#!/usr/bin/env python3
"""How large an SNN can a learned policy actually optimize?

Every attempt in this repository to teach a policy network to optimize a spiking
net was run at one fixed, large scale (784-256-256-256-256, ~400k weights, on
KMNIST) and every one of them failed. That tells us the policy failed; it does
not tell us *why*, because a single failure at one scale cannot distinguish
"policies fundamentally cannot optimize spiking nets" from "policies cannot
optimize nets THIS big".

This tool answers that by measuring the boundary. It starts with an SNN so small
the policy ought to be able to optimize it exactly, and grows the network -- width
first, then depth -- until the policy stops keeping up. The output is a bound.

The task is synthetic on purpose: labels come from a fixed random nonlinear
teacher, so they are deterministic (the achievable ceiling is 100%, with no label
noise to hide behind) and the difficulty is a knob rather than a mystery.

At every size we measure four points, all with the same frozen-features + ridge
readout protocol:

    untrained   the random SNN            -- the floor
    stdp        the hand-designed rule    -- the unsupervised reference
    teacher     surrogate-gradient descent-- the optimization CEILING
    policy      the learned optimizer     -- what we are actually measuring

and report the normalised optimisation gap

    gap = (teacher - policy) / (teacher - untrained)

which is 0 when the policy matches gradient descent and 1 when it has learned
nothing. The scale at which `gap` departs from 0 is the answer.
"""

import argparse
import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Synthetic task: a fixed random nonlinear teacher. Labels are deterministic,  #
# so the ceiling is 100% and any shortfall is the optimizer's fault, not the   #
# data's.                                                                      #
# --------------------------------------------------------------------------- #
def make_task(opt, device):
    """Multi-cluster classification: each class is the union of several Gaussian
    blobs scattered through input space.

    This shape is chosen deliberately. A linear model is near-helpless on it (a
    class is several disconnected blobs, so no hyperplane carves it out), while an
    MLP solves it easily -- so the accuracy a model reaches is a direct measure of
    how good its LEARNED FEATURES are, which is exactly what an optimizer is being
    judged on. Difficulty is two honest knobs: clusters per class and blob noise.

    (The first version of this task was a random tanh teacher. It turned out to be
    ~linear -- ridge on the raw inputs already scored 65.3%, above the optimally
    trained SNN -- so it measured nothing. Always check that the task needs the
    features you claim to be studying.)
    """
    generator = torch.Generator(device=device).manual_seed(opt.task_seed)
    count = opt.classes * opt.clusters
    centres = torch.randn((count, opt.input_dim), device=device, generator=generator)
    centres = centres / centres.norm(dim=1, keepdim=True) * opt.spread
    labels = torch.arange(opt.classes, device=device).repeat_interleave(opt.clusters)

    def sample(n, seed):
        gen = torch.Generator(device=device).manual_seed(seed)
        which = torch.randint(0, count, (n,), device=device, generator=gen)
        x = centres[which] + opt.noise * torch.randn((n, opt.input_dim), device=device, generator=gen)
        lo, hi = x.min(dim=0, keepdim=True).values, x.max(dim=0, keepdim=True).values
        return (x - lo) / (hi - lo).clamp_min(1e-6), labels[which]

    return sample(opt.train_samples, opt.task_seed + 1), sample(opt.test_samples, opt.task_seed + 2)


# --------------------------------------------------------------------------- #
# A spiking net of configurable width and depth                                #
# --------------------------------------------------------------------------- #
class Net:
    def __init__(self, layers, width, input_dim, timesteps=12, beta=0.9,
                 weight_norm=2.0, threshold=1.0, alpha=2.0):
        self.layers = layers
        self.width = width
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.beta = beta
        self.weight_norm = weight_norm
        self.threshold = threshold
        self.alpha = alpha
        self.top_k = max(1, width // 4)

    def fan_in(self, index):
        return self.input_dim if index == 0 else self.width

    def feature_dim(self):
        return self.layers * self.width


def init_net(net, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    weights, thresholds = [], []
    for index in range(net.layers):
        fan_in = net.fan_in(index)
        basis = torch.randn((max(fan_in, net.width), min(fan_in, net.width)),
                            device=device, generator=generator)
        q, _ = torch.linalg.qr(basis, mode="reduced")
        weight = q[:fan_in, :net.width] if fan_in >= net.width else q[:net.width, :fan_in].t()
        weights.append(weight.t().contiguous()[:net.width, :fan_in].clone())
        weights[-1] = normalize_rows(weights[-1], net.weight_norm)
        thresholds.append(torch.full((net.width,), net.threshold, device=device))
    return weights, thresholds


def normalize_rows(matrix, norm):
    """`norm` may be a scalar or a per-row tensor -- the policy sets its own scale."""
    matrix = matrix - matrix.mean(dim=-1, keepdim=True)
    if torch.is_tensor(norm) and norm.dim() == 1:
        norm = norm.unsqueeze(-1)
    return matrix * (norm / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-8))


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centered, alpha):
        ctx.save_for_backward(centered)
        ctx.alpha = alpha
        return (centered >= 0).to(centered.dtype)

    @staticmethod
    def backward(ctx, grad):
        (centered,) = ctx.saved_tensors
        return grad / (1.0 + (ctx.alpha * centered) ** 2), None


def run(weights, thresholds, x, net, differentiable=False, probes=None, collect=False):
    """Forward. Returns per-layer readout features, and (if collect) the local
    statistics a plasticity rule sees: per-sample pre/post rates."""
    batch = x.shape[0]
    device = x.device
    voltage = [torch.zeros((batch, net.width), device=device) for _ in weights]
    previous = [torch.zeros_like(v) for v in voltage]
    filtered = [torch.zeros_like(v) for v in voltage]
    summed = [torch.zeros_like(v) for v in voltage]
    rates = [torch.zeros_like(v) for v in voltage]

    for _ in range(net.timesteps):
        signal = x
        for index, weight in enumerate(weights):
            drive = voltage[index] * net.beta + signal @ weight.t()
            membrane = drive - previous[index] * thresholds[index]
            score = membrane - thresholds[index]
            if probes is not None:
                score = score + probes[index]
            _, winners = score.detach().topk(net.top_k, dim=1, sorted=False)
            mask = torch.zeros_like(score).scatter_(1, winners, 1.0)
            spike = mask * (Spike.apply(score, net.alpha) if differentiable
                            else (score >= 0).to(score.dtype))
            voltage[index] = membrane
            previous[index] = spike
            filtered[index] = filtered[index] * net.beta + spike
            summed[index] = summed[index] + filtered[index]
            if collect:
                rates[index] = rates[index] + spike
            signal = spike

    features = [s / net.timesteps for s in summed]
    if not collect:
        return features, None
    post_rates = [r / net.timesteps for r in rates]
    return features, post_rates


# --------------------------------------------------------------------------- #
# Readout                                                                      #
# --------------------------------------------------------------------------- #
def fit_ridge(features, labels, classes, ridge=1.0):
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(0.02)
    design = torch.cat(((features - mean) / std,
                        torch.ones((len(features), 1), device=features.device)), dim=1)
    gram = design.t() @ design
    gram.diagonal().add_(ridge)
    return mean, std, torch.linalg.solve(gram, design.t() @ F.one_hot(labels, classes).float())


def readout(features, mean, std, matrix):
    design = torch.cat(((features - mean) / std,
                        torch.ones((len(features), 1), device=features.device)), dim=1)
    return design @ matrix


@torch.no_grad()
def evaluate(weights, thresholds, net, train, test, classes):
    train_x, train_y = train
    test_x, test_y = test
    tr, _ = run(weights, thresholds, train_x, net)
    te, _ = run(weights, thresholds, test_x, net)
    mean, std, matrix = fit_ridge(torch.cat(tr, dim=1), train_y, classes)
    logits = readout(torch.cat(te, dim=1), mean, std, matrix)
    return logits.argmax(dim=1).eq(test_y).float().mean().item()


def head_from_ridge(weights, thresholds, net, train, classes, device):
    """A frozen linear head, so the teacher and the policy have a loss to descend."""
    with torch.no_grad():
        features, _ = run(weights, thresholds, train[0], net)
        mean, std, matrix = fit_ridge(torch.cat(features, dim=1), train[1], classes)

    class Head(nn.Module):
        def forward(self, concat):
            return readout(concat, mean, std, matrix)

    return Head().to(device)


# --------------------------------------------------------------------------- #
# Credit: per-sample, per-neuron error dL/dprobe -- the one signal that ever   #
# worked in the KMNIST experiments.                                            #
# --------------------------------------------------------------------------- #
def credit(weights, thresholds, x, y, head, net):
    with torch.enable_grad():
        leaves = [w.detach().clone().requires_grad_(True) for w in weights]
        probes = [torch.zeros((x.shape[0], net.width), device=x.device, requires_grad=True)
                  for _ in weights]
        features, _ = run(leaves, thresholds, x, net, differentiable=True, probes=probes)
        loss = F.cross_entropy(head(torch.cat(features, dim=1)), y)
        grads = torch.autograd.grad(loss, leaves + probes)
    n = len(weights)
    return [g.detach() for g in grads[:n]], [g.detach() for g in grads[n:]], loss.detach()


def apply_update(weight, direction, step, norm):
    """Rotate each neuron's incoming weights by its OWN step size, then renormalise
    each row to its OWN target norm.

    Normalisation is not bureaucracy here -- it is what keeps the LIF/top-k
    dynamics alive (weight scale has to stay matched to threshold scale; let it
    drift and the net either fires for everything or nothing). Removing it
    outright collapsed the policy to chance and cost the teacher 10 points. So the
    manifold stays, and the policy is handed every knob ON it: per-neuron step
    size, per-neuron norm, and the thresholds themselves."""
    moved = weight + step.unsqueeze(-1) * direction
    return normalize_rows(moved, norm)


# --------------------------------------------------------------------------- #
# Reference optimizers                                                         #
# --------------------------------------------------------------------------- #
def train_teacher(net, train, test, classes, opt, device, seed):
    """The optimization CEILING, given every advantage.

    Proper end-to-end surrogate BPTT: Adam over the weights, the thresholds AND a
    jointly-trained readout head (not a stale frozen ridge -- chasing one cripples
    the optimizer). Swept over learning rates, best result kept. If the policy is
    going to be called better than gradient descent, gradient descent has to have
    been given its best shot.
    """
    train_x, train_y = train
    best_score, best_state = -1.0, None
    for lr in opt.teacher_lrs:
        weights, thresholds = init_net(net, device, seed)
        weights = [w.clone().requires_grad_(True) for w in weights]
        thresholds = [t.clone().requires_grad_(True) for t in thresholds]
        head = nn.Linear(net.feature_dim(), classes).to(device)
        optimiser = torch.optim.Adam(list(head.parameters()) + weights + thresholds, lr=lr)
        for epoch in range(opt.epochs):
            order = torch.randperm(len(train_x), device=device)
            for start in range(0, len(order), opt.batch):
                idx = order[start:start + opt.batch]
                features, _ = run(weights, thresholds, train_x[idx], net, differentiable=True)
                loss = F.cross_entropy(head(torch.cat(features, dim=1)), train_y[idx])
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                with torch.no_grad():   # stay on the manifold the dynamics need
                    for w in weights:
                        w.copy_(normalize_rows(w, net.weight_norm))
                    for t in thresholds:
                        t.clamp_(0.1, 5.0)
        frozen = ([w.detach() for w in weights], [t.detach() for t in thresholds])
        acc = evaluate(frozen[0], frozen[1], net, train, test, classes)
        if acc > best_score:
            best_score, best_state = acc, frozen
    return best_state


def train_stdp(weights, thresholds, net, train, opt, device):
    train_x = train[0]
    decay, ltd = 0.5, 0.5
    for epoch in range(opt.epochs):
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), opt.batch):
            x = train_x[order[start:start + opt.batch]]
            count = len(x)
            voltage = [torch.zeros((count, net.width), device=device) for _ in weights]
            previous = [torch.zeros_like(v) for v in voltage]
            pre_trace = [torch.zeros((count, net.fan_in(i)), device=device) for i in range(net.layers)]
            post_trace = [torch.zeros_like(v) for v in voltage]
            pot = [torch.zeros_like(w) for w in weights]
            dep = [torch.zeros_like(w) for w in weights]
            seen = [torch.zeros(net.width, device=device) for _ in weights]
            with torch.no_grad():
                for _ in range(net.timesteps):
                    signal = x
                    for i, weight in enumerate(weights):
                        pre = signal
                        drive = voltage[i] * net.beta + pre @ weight.t()
                        membrane = drive - previous[i] * thresholds[i]
                        score = membrane - thresholds[i]
                        _, winners = score.topk(net.top_k, dim=1, sorted=False)
                        mask = torch.zeros_like(score).scatter_(1, winners, 1.0)
                        spike = mask * (score >= 0).float()
                        voltage[i] = membrane
                        pre_trace[i].mul_(decay).add_(pre)
                        pot[i].addmm_(spike.t(), pre_trace[i])
                        dep[i].addmm_(post_trace[i].t(), pre)
                        post_trace[i].mul_(decay).add_(spike)
                        previous[i] = spike
                        seen[i].add_(spike.sum(dim=0))
                        signal = spike
                for i in range(net.layers):
                    e = pot[i].add(dep[i], alpha=-ltd)
                    e.div_(seen[i].unsqueeze(1).clamp_min_(1.0))
                    e.sub_(e.mean(dim=1, keepdim=True))
                    e.div_(e.norm(dim=1, keepdim=True).clamp_min_(1e-8))
                    weights[i] = normalize_rows(weights[i] + opt.stdp_lr * e, net.weight_norm)
    return weights, thresholds


# --------------------------------------------------------------------------- #
# The learned policy optimizer                                                 #
# --------------------------------------------------------------------------- #
class Policy(nn.Module):
    """The policy IS the optimizer. Maximal action space.

    It sees the per-sample credit, the activity, its OWN current weights, and its
    OWN accumulated state (momentum and second moment). It emits a real-valued
    weight update *with magnitude* -- it sets its own per-synapse step size -- and
    a threshold update. Nothing is renormalised away.

    This is strictly more expressive than Adam: the per-synapse head is handed
    `m / sqrt(v)` directly, so it can reproduce the Adam update exactly and then
    depart from it. It can also do what no gradient method can -- condition the
    step on the current weight (decay, saturation) and retune thresholds.
    """

    def __init__(self, width=256, depth=2, heads=8, syn_width=64, use_gradient=False):
        super().__init__()
        def mlp(in_dim, out_dim):
            layers = [nn.Linear(in_dim, width), nn.GELU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.GELU()]
            return nn.Sequential(*layers, nn.Linear(width, out_dim))
        self.post = mlp(4, heads)     # [error, |error|, post_rate, threshold]
        self.pre = mlp(3, heads)      # [pre_rate, sqrt, square]
        # Per-synapse head: gradient-shaped proposal + weight + optimizer state.
        self.use_gradient = use_gradient
        self.syn = nn.Sequential(
            nn.Linear(8 if use_gradient else 6, syn_width), nn.GELU(),
            nn.Linear(syn_width, syn_width), nn.GELU(),
            nn.Linear(syn_width, 1))
        # Per-neuron knobs: its own step size, its own weight norm, its own threshold.
        self.neuron = mlp(4, 3)
        self.beta1, self.beta2 = 0.9, 0.999
        for module in (self.syn[-1], self.neuron[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, weight, error, post_rate, threshold, pre_rate, state, lr, base_norm,
                gradient=None, count=1):
        batch = error.shape[0]
        momentum, second = state
        thr = threshold.unsqueeze(0).expand(batch, -1)
        post_feat = standardize(torch.stack([error, error.abs(), post_rate, thr], -1))
        pre_feat = standardize(torch.stack([pre_rate, pre_rate.sqrt(), pre_rate ** 2], -1))
        c_post = self.post(post_feat)                            # [B, post, H]
        c_pre = self.pre(pre_feat)                               # [B, pre,  H]
        proposal = torch.einsum("bph,bqh->pq", c_post, c_pre)    # gradient-shaped [post, pre]

        # Whose signal do we accumulate: the true descent direction, or the policy's
        # own learned proposal?
        signal = -gradient if self.use_gradient else proposal
        new_m = self.beta1 * momentum + (1 - self.beta1) * signal
        new_v = self.beta2 * second + (1 - self.beta2) * signal ** 2
        # Bias correction. Omitting it (the original bug) makes the first steps ~3x
        # too large, because m and v both start biased toward zero.
        t = float(count)
        m_hat = new_m / (1 - self.beta1 ** t)
        v_hat = new_v / (1 - self.beta2 ** t)
        adam = m_hat / (v_hat.sqrt() + 1e-8)                     # a real Adam update

        parts = [proposal, weight, new_m, new_v.sqrt(), adam, torch.sign(signal)]
        if self.use_gradient:
            parts += [signal, signal / signal.abs().mean().clamp_min(1e-8)]
        channels = torch.stack(parts, dim=-1)

        # THE policy IS ADAM AT INITIALISATION, plus a zero-initialised residual.
        # Previously the output head was zero-init, so the policy emitted *no update at
        # all* and had to rediscover the entire concept of gradient descent through
        # meta-gradients -- while the row-normalisation in apply_update stripped the
        # magnitude, so it could not even express Adam. Starting from a known-good
        # optimizer means meta-training only has to IMPROVE on it, never invent it.
        residual = self.syn(channels).squeeze(-1)
        direction = adam + residual
        self.last_residual = residual

        summary = standardize(torch.stack(
            [error.mean(0), error.abs().mean(0), post_rate.mean(0), threshold], -1).unsqueeze(0))
        knobs = self.neuron(summary).squeeze(0)                  # [post, 3]
        step = lr * torch.exp(knobs[:, 0].clamp(-3, 3))          # adaptive per-neuron step
        norm = base_norm * torch.exp(knobs[:, 1].clamp(-1, 1))   # adaptive per-neuron scale
        d_thresh = lr * knobs[:, 2]
        # Deviation from plain Adam = the direction residual AND every knob. Penalising
        # only the residual (the first attempt) left the step size, the weight norm and
        # the thresholds free to blow up, which is what actually destroyed the policy.
        self.deviation = residual.pow(2).mean() + knobs.pow(2).mean()
        return direction, step, norm, d_thresh, (new_m.detach(), new_v.detach())

    def zero_state(self, net, device):
        return ([torch.zeros((net.width, net.fan_in(i)), device=device) for i in range(net.layers)],
                [torch.zeros((net.width, net.fan_in(i)), device=device) for i in range(net.layers)])

    def count(self):
        return sum(p.numel() for p in self.parameters())


def standardize(t):
    mean = t.mean(dim=(0, 1), keepdim=True)
    std = t.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
    return (t - mean) / std


def policy_step(policy, weights, thresholds, x, y, head, net, state, lr, count=1):
    """One optimizer step: credit -> policy -> weight AND threshold update."""
    with torch.no_grad():
        _, post_rates = run(weights, thresholds, x, net, collect=True)
    grads, errors, _ = credit(weights, thresholds, x, y, head, net)
    momentum, second = state
    new_w, new_t, new_m, new_v = [], [], [], []
    for i in range(net.layers):
        pre_rate = x if i == 0 else post_rates[i - 1]
        direction, step, norm, d_thresh, (m, v) = policy(
            weights[i], errors[i], post_rates[i], thresholds[i], pre_rate,
            (momentum[i], second[i]), lr, net.weight_norm,
            gradient=grads[i] if policy.use_gradient else None, count=count)
        new_w.append(apply_update(weights[i], direction, step, norm))
        new_t.append((thresholds[i] + d_thresh).clamp(0.1, 5.0))
        new_m.append(m)
        new_v.append(v)
    return new_w, new_t, (new_m, new_v)


def train_policy(policy, net, train, classes, opt, device, seed):
    if opt.meta_steps == 0:      # deploy the Adam-initialised policy untouched
        return policy
    """Meta-train: take the policy's step, then minimise the loss it leaves behind
    on a held-out query batch. The optimizee walks its own trajectory."""
    train_x, train_y = train
    optimiser = torch.optim.Adam(policy.parameters(), lr=opt.meta_lr)
    step = 0
    while step < opt.meta_steps:
        weights, thresholds = init_net(net, device, seed * 7919 + step)
        state = policy.zero_state(net, device)
        head = head_from_ridge(weights, thresholds, net, train, classes, device)
        for local in range(opt.horizon):
            if step >= opt.meta_steps:
                break
            # The features move as the policy edits them, so the readout that defines
            # the loss has to move with them. Fitting it once on the initial random
            # weights (the original bug) means optimising a loss for a network that
            # no longer exists -- and deployment refits, so it was a train/deploy
            # mismatch too.
            if local % opt.head_refit == 0 and local > 0:
                head = head_from_ridge(weights, thresholds, net, train, classes, device)
            # Unroll K policy steps before scoring. A single-step objective teaches a
            # MYOPIC rule -- good for one update, useless over the thousands of updates
            # it must actually run. Scoring only after K steps is what makes the policy
            # optimise a TRAJECTORY instead of a step.
            edited, new_t = weights, thresholds
            penalty = 0.0
            for u in range(opt.unroll):
                idx = torch.randint(0, len(train_x), (opt.batch,), device=device)
                edited, new_t, state = policy_step(policy, edited, new_t,
                                                   train_x[idx], train_y[idx], head, net,
                                                   state, opt.inner_lr, count=step + u + 1)
                penalty = penalty + policy.deviation
            qidx = torch.randint(0, len(train_x), (opt.batch,), device=device)
            features, _ = run(edited, new_t, train_x[qidx], net, differentiable=True)
            # TRUST REGION. The meta-loss scores the network after `unroll` steps, but
            # deployment runs ~2800 -- so an unconstrained meta-objective rewards
            # aggressive short-horizon greed, which wrecks the long trajectory. That is
            # why meta-training made the policy monotonically WORSE than the Adam it
            # started as. Penalising the residual keeps the policy near a known-good
            # optimizer, so meta-training can only buy small, genuinely earned gains.
            loss = F.cross_entropy(head(torch.cat(features, dim=1)), train_y[qidx])
            loss = loss + opt.trust * penalty / max(1, opt.unroll)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            weights = [w.detach() for w in edited]
            thresholds = [t.detach() for t in new_t]
            step += 1
    return policy


def deploy_policy(policy, net, train, classes, opt, device, seed):
    """The policy optimizes the SNN. The readout is trained jointly by gradient
    descent -- exactly as the teacher does it. Using a frozen ridge readout here
    instead (the earlier version) cost ~15 points and made an Adam-initialised
    policy look far worse than the Adam it was a copy of."""
    train_x, train_y = train
    weights, thresholds = init_net(net, device, seed)
    state = policy.zero_state(net, device)
    head = nn.Linear(net.feature_dim(), classes).to(device)
    head_opt = torch.optim.Adam(head.parameters(), lr=opt.head_lr)
    taken = 0
    for epoch in range(opt.epochs):
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), opt.batch):
            idx = order[start:start + opt.batch]
            x, y = train_x[idx], train_y[idx]
            # train the readout on the current features
            features, _ = run([w.detach() for w in weights], [t.detach() for t in thresholds],
                              x, net, differentiable=False)
            loss = F.cross_entropy(head(torch.cat(features, dim=1).detach()), y)
            head_opt.zero_grad(set_to_none=True)
            loss.backward()
            head_opt.step()
            with torch.no_grad():
                weights, thresholds, state = policy_step(
                    policy, weights, thresholds, x, y, head, net,
                    state, opt.inner_lr, count=taken + 1)
            taken += 1
    return weights, thresholds


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--input-dim", type=int, default=16)
    p.add_argument("--classes", type=int, default=5)
    p.add_argument("--clusters", type=int, default=20, help="Gaussian blobs per class (nonlinearity knob)")
    p.add_argument("--spread", type=float, default=3.0, help="how far apart the blobs sit")
    p.add_argument("--noise", type=float, default=0.35, help="blob width (difficulty knob)")
    p.add_argument("--task-seed", type=int, default=7)
    p.add_argument("--train-samples", type=int, default=12000)
    p.add_argument("--test-samples", type=int, default=4000)
    p.add_argument("--sizes", default="1x4,1x8,1x16,1x32,1x64,1x128,2x64,3x64,4x64",
                   help="comma list of LAYERSxWIDTH to sweep")
    p.add_argument("--epochs", type=int, default=15,
                   help="optimization budget -- IDENTICAL for teacher, stdp and policy")
    p.add_argument("--teacher-lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2],
                   help="teacher is swept over these and the BEST is kept -- a real ceiling")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--stdp-lr", type=float, default=0.005)
    p.add_argument("--meta-steps", type=int, default=600)
    p.add_argument("--meta-lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=3e-3)
    p.add_argument("--trust", type=float, default=0.0,
                   help="penalty keeping the policy near Adam (0 = unconstrained, which DEGRADES it)")
    p.add_argument("--horizon", type=int, default=250,
                   help="meta-training episode length; deployment runs ~2800 steps, so short\n                        episodes mean the policy never sees the trajectory it must run")
    p.add_argument("--head-refit", type=int, default=25,
                   help="refit the readout every N optimizer steps (train AND deploy)")
    p.add_argument("--unroll", type=int, default=1,
                   help="policy steps taken before the meta-loss is scored (1 = myopic)")
    p.add_argument("--policy-width", type=int, default=256)
    p.add_argument("--policy-depth", type=int, default=2)
    p.add_argument("--use-gradient", action="store_true",
                   help="give the policy the true gradient -- learn to IMPROVE a descent\n                        direction rather than rediscover one from local signals")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed0", type=int, default=1)
    return p.parse_args()


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("snn_scaling: CUDA required")
    device = torch.device(opt.device)
    torch.cuda.set_device(device)

    train, test = make_task(opt, device)
    print(f"synthetic task: {opt.input_dim}-dim inputs, {opt.classes} classes from a fixed "
          f"random nonlinear teacher (labels deterministic, ceiling 100%)")
    print(f"{len(train[0])} train / {len(test[0])} test, device={torch.cuda.get_device_name(device)}")
    print("references on this task: chance 20%  |  linear-on-raw ~42%  |  MLP-on-raw ~99%")
    print("=> accuracy above ~42% is entirely earned by LEARNED SPIKING FEATURES\n")
    print(f"{'size':>8} {'params':>8} {'untrained':>10} {'stdp':>8} {'teacher':>9} "
          f"{'policy':>8} {'gap':>7}")
    print("-" * 64)

    for spec in opt.sizes.split(","):
        layers, width = (int(v) for v in spec.lower().split("x"))
        net = Net(layers, width, opt.input_dim)
        params = sum(net.width * net.fan_in(i) for i in range(layers))

        untrained, stdp, teacher, learned = [], [], [], []
        for repeat in range(opt.seeds):
            seed = opt.seed0 + repeat
            w0, t0 = init_net(net, device, seed)
            untrained.append(evaluate(w0, t0, net, train, test, opt.classes))

            w, t = init_net(net, device, seed)
            w, t = train_stdp(w, t, net, train, opt, device)
            stdp.append(evaluate(w, t, net, train, test, opt.classes))

            w, t = train_teacher(net, train, test, opt.classes, opt, device, seed)
            teacher.append(evaluate(w, t, net, train, test, opt.classes))

            policy = Policy(opt.policy_width, opt.policy_depth,
                            use_gradient=opt.use_gradient).to(device)
            policy = train_policy(policy, net, train, opt.classes, opt, device, seed)
            w, t = deploy_policy(policy, net, train, opt.classes, opt, device, seed)
            learned.append(evaluate(w, t, net, train, test, opt.classes))

        u, s, tc, pl = (100 * sum(v) / len(v) for v in (untrained, stdp, teacher, learned))
        gap = (tc - pl) / (tc - u) if tc > u else float("nan")
        print(f"{spec:>8} {params:>8,} {u:>9.2f}% {s:>7.2f}% {tc:>8.2f}% {pl:>7.2f}% {gap:>7.2f}",
              flush=True)


if __name__ == "__main__":
    main()
