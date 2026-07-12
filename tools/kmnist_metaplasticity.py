#!/usr/bin/env python3
"""Meta-learn a local plasticity rule that wires a KMNIST SNN better than STDP.

A large policy network (>=50M params) reads *local* signals at each synapse --
presynaptic rate, postsynaptic rate/threshold, current weight, causal and
anti-causal spike correlations, and (in three-factor mode) a per-neuron top-down
error modulator -- and emits a weight-update direction for the same
784-256-256-256-256 LIF network the STDP experiment (tools/kmnist_stdp.py) uses.

The policy is trained by GRADIENT IMITATION: at each inner step the surrogate
gradient of the readout loss w.r.t. the SNN weights is computed (this is exactly
"how a weight change affects the loss"), and the policy is trained so its
per-synapse-local update direction aligns with that global gradient. The learned
rule reads only local signals; the teacher that trains it is global. This is the
only signal that scales to a 50-100M-param policy -- evolution/RL variance
explodes at that dimensionality.

The third factor is the true per-neuron error dL/du, obtained as the gradient
w.r.t. a per-neuron score probe -- the same backprop that yields the target, so
the target is predictable from the inputs. It is a top-down modulator (its
computation uses the readout, i.e. weight transport), which is what makes the
three-factor rule loss-aware; the two-factor rule sees no error and must rely on
STDP's unsupervised signals alone.

Benchmark ladder (all frozen -> ridge readout, identical to the STDP protocol):
STDP (hand-designed, unsupervised) -> learned two-factor rule (STDP's information,
learned) -> learned three-factor rule (adds the top-down error) -> surrogate-GD
teacher (global upper bound). CUDA only, like the other backends.
"""

import argparse
import csv
import gzip
import math
import os
import random
import struct
import time

# cuBLAS needs this for deterministic GEMMs; must precede the first CUDA op.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 4
CLASSES = 10


def fail(message):
    raise SystemExit(f"kmnist_metaplasticity: {message}")


# --------------------------------------------------------------------------- #
# Data (same IDX loader as tools/kmnist_stdp.py)                              #
# --------------------------------------------------------------------------- #
def read_idx(path, expected_magic):
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            fail(f"short IDX header in {path}")
        magic, count = struct.unpack(">II", header)
        if magic != expected_magic:
            fail(f"bad IDX magic in {path}")
        if expected_magic == 0x803:
            geometry = stream.read(8)
            if len(geometry) != 8 or struct.unpack(">II", geometry) != (28, 28):
                fail(f"IDX images in {path} are not 28x28")
        data = stream.read()
    stride = INPUT_SIZE if expected_magic == 0x803 else 1
    if len(data) != count * stride:
        fail(f"short IDX payload in {path}")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(count, stride), count


def load_dataset(data_dir, device, train_limit=0, test_limit=0):
    def load(images_name, labels_name, limit):
        images, count = read_idx(os.path.join(data_dir, images_name), 0x803)
        labels, label_count = read_idx(os.path.join(data_dir, labels_name), 0x801)
        if count != label_count:
            fail("image/label count mismatch")
        if limit:
            count = min(count, limit)
            images = images[:count]
            labels = labels[:count]
        return images.to(device), labels.view(-1).to(device).long()

    train = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", train_limit)
    test = load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", test_limit)
    return train, test


# --------------------------------------------------------------------------- #
# Surrogate spike (atan, peak-normalized so phi(0)=1, like the repo's BPTT)   #
# --------------------------------------------------------------------------- #
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centered, alpha):
        ctx.save_for_backward(centered)
        ctx.alpha = alpha
        return (centered >= 0).to(centered.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (centered,) = ctx.saved_tensors
        phi = 1.0 / (1.0 + (ctx.alpha * centered) ** 2)
        return grad_output * phi, None


def surrogate_spike(centered, alpha):
    return SurrogateSpike.apply(centered, alpha)


def soft_spike(centered, alpha):
    """Fully smooth spike proxy for gradient checking (real autograd, no custom
    backward), so finite differences of the forward match the analytic gradient."""
    return 0.5 + (1.0 / math.pi) * torch.atan(alpha * centered)


# --------------------------------------------------------------------------- #
# SNN: weights [post, pre], one threshold per post neuron. QR init matches    #
# the STDP tool so the two rules start from the same distribution.            #
# --------------------------------------------------------------------------- #
def fan_in_of(layer_index):
    return INPUT_SIZE if layer_index == 0 else HIDDEN_SIZE


def init_snn(device, generator, weight_norm, initial_threshold):
    weights, thresholds = [], []
    for index in range(HIDDEN_LAYERS):
        fan_in = fan_in_of(index)
        basis = torch.randn((fan_in, HIDDEN_SIZE), device=device, generator=generator)
        orthogonal, triangular = torch.linalg.qr(basis, mode="reduced")
        sign = triangular.diagonal().sign().masked_fill_(triangular.diagonal().eq(0), 1.0)
        weight = (orthogonal * sign).t().contiguous().mul_(weight_norm)
        weights.append(weight)
        thresholds.append(torch.full((HIDDEN_SIZE,), initial_threshold, device=device))
    return weights, thresholds


def normalize_rows(matrix, norm):
    """Mean-center and L2-normalize each post row, then rescale to `norm`."""
    matrix = matrix - matrix.mean(dim=1, keepdim=True)
    return matrix * (norm / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8))


class SNNConfig:
    def __init__(self, timesteps=20, beta=0.95, top_k=32, trace_decay=0.5,
                 alpha=2.0, weight_norm=2.0, initial_threshold=1.0):
        self.timesteps = timesteps
        self.beta = beta
        self.top_k = top_k
        self.trace_decay = trace_decay
        self.alpha = alpha
        self.weight_norm = weight_norm
        self.initial_threshold = initial_threshold


def winner_mask(score, top_k):
    _, index = score.topk(top_k, dim=1, sorted=False)
    mask = torch.zeros_like(score)
    mask.scatter_(1, index, 1.0)
    return mask


def run_snn(weights, thresholds, pixels, config, spike_mode, collect, probes=None):
    """Forward the frozen hidden stack.

    spike_mode: 'hard' (exact step), 'surrogate' (hard forward, atan-surrogate
    backward), or 'soft' (smooth primitive, for gradient checking). Returns
    per-layer filtered-spike features [B, HIDDEN] (differentiable in the
    surrogate/soft modes) and, when `collect`, the local plasticity statistics
    the policy consumes. `probes`, if given, is a per-layer [HIDDEN] tensor added
    to each neuron's score every step; dL/dprobe is the true per-neuron top-down
    error (the third factor), consistent with the gradient target.
    """
    batch = pixels.shape[0]
    device = pixels.device
    dtype = weights[0].dtype
    voltages = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]
    previous = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]
    filtered = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]
    feature_sum = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]

    if collect:
        potentiation = [torch.zeros_like(w) for w in weights]
        depression = [torch.zeros_like(w) for w in weights]
        pre_traces = [torch.zeros((batch, fan_in_of(i)), device=device, dtype=dtype) for i in range(HIDDEN_LAYERS)]
        post_traces = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]
        pre_activity = [torch.zeros(fan_in_of(i), device=device, dtype=dtype) for i in range(HIDDEN_LAYERS)]
        post_activity = [torch.zeros(HIDDEN_SIZE, device=device, dtype=dtype) for _ in weights]

    for _ in range(config.timesteps):
        signal = pixels
        for index, weight in enumerate(weights):
            pre = signal
            drive = voltages[index] * config.beta + pre @ weight.t()
            voltage = drive - previous[index] * thresholds[index]
            score = voltage - thresholds[index]
            if probes is not None:
                score = score + probes[index]
            mask = winner_mask(score.detach(), config.top_k)
            if spike_mode == "surrogate":
                spike = mask * surrogate_spike(score, config.alpha)
            elif spike_mode == "soft":
                spike = mask * soft_spike(score, config.alpha)
            else:
                spike = mask * (score >= 0).to(score.dtype)
            voltages[index] = voltage
            previous[index] = spike
            filtered[index] = filtered[index] * config.beta + spike
            feature_sum[index] = feature_sum[index] + filtered[index]
            if collect:
                pre_traces[index] = pre_traces[index] * config.trace_decay + pre
                potentiation[index] = potentiation[index] + spike.t() @ pre_traces[index]
                depression[index] = depression[index] + post_traces[index].t() @ pre
                post_traces[index] = post_traces[index] * config.trace_decay + spike
                pre_activity[index] = pre_activity[index] + pre.sum(dim=0)
                post_activity[index] = post_activity[index] + spike.sum(dim=0)
            signal = spike

    features = [total / config.timesteps for total in feature_sum]
    if not collect:
        return features, None

    denom = batch * config.timesteps
    stats = []
    for index in range(HIDDEN_LAYERS):
        spikes_per_post = post_activity[index].clamp_min(1.0).unsqueeze(1)
        stats.append({
            "potentiation": potentiation[index] / spikes_per_post,
            "depression": depression[index] / spikes_per_post,
            "pre_rate": pre_activity[index] / denom,       # [pre]
            "post_rate": post_activity[index] / denom,     # [post]
        })
    return features, stats


# --------------------------------------------------------------------------- #
# Ridge readout (same as the STDP tool, for an apples-to-apples benchmark)    #
# --------------------------------------------------------------------------- #
def fit_ridge(features, labels, ridge):
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(0.02)
    standardized = (features - mean) / std
    design = torch.cat((standardized, torch.ones((len(features), 1), device=features.device)), dim=1)
    targets = F.one_hot(labels, CLASSES).float()
    gram = design.t() @ design
    gram.diagonal().add_(ridge)
    readout = torch.linalg.solve(gram, design.t() @ targets)
    return mean, std, readout


def ridge_scores(features, labels, mean, std, readout):
    design = torch.cat(((features - mean) / std, torch.ones((len(features), 1), device=features.device)), dim=1)
    logits = design @ readout
    loss = F.cross_entropy(logits, labels).item()
    accuracy = logits.argmax(dim=1).eq(labels).float().mean().item()
    return loss, accuracy


# --------------------------------------------------------------------------- #
# The plasticity policy: big per-neuron encoders, cheap per-synapse combiner. #
# --------------------------------------------------------------------------- #
PRE_FEATURES = 3     # [pre_rate, pre_rate^2, sqrt(pre_rate)]
POST_FEATURES = 4    # [post_rate, threshold, error_mod, post_rate^2]
SYN_FEATURES = 4     # [weight, potentiation, depression, |weight|]


def standardize(features, dims):
    """Zero-mean, unit-std per feature channel across the given neuron dim(s)."""
    mean = features.mean(dim=dims, keepdim=True)
    std = features.std(dim=dims, keepdim=True).clamp_min(1e-5)
    return (features - mean) / std


def mlp(in_dim, width, depth, out_dim):
    layers = [nn.Linear(in_dim, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class PlasticityPolicy(nn.Module):
    def __init__(self, width=4096, depth=3, heads=8, head_dim=48, gate_width=64,
                 use_momentum=False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.use_momentum = use_momentum
        syn_features = SYN_FEATURES + (1 if use_momentum else 0)
        embed = heads * head_dim
        self.pre_encoder = mlp(PRE_FEATURES, width, depth, embed)
        self.post_encoder = mlp(POST_FEATURES, width, depth, embed)
        self.gate = nn.Sequential(
            nn.Linear(syn_features, gate_width), nn.GELU(),
            nn.Linear(gate_width, gate_width), nn.GELU(),
            nn.Linear(gate_width, heads + 1),   # per-head gate + a bias channel
        )
        # Scale bilinear scores into a sane range before gating.
        self.score_scale = 1.0 / math.sqrt(head_dim)

    def forward(self, weight, stats, error_mod, momentum=None):
        post = weight.shape[0]
        pre = weight.shape[1]
        pre_rate = stats["pre_rate"].clamp_min(0.0)
        post_rate = stats["post_rate"].clamp_min(0.0)
        pre_feats = torch.stack([pre_rate, pre_rate ** 2, pre_rate.sqrt()], dim=1)
        post_feats = torch.stack([post_rate, stats["threshold"], error_mod, post_rate ** 2], dim=1)
        # Standardize each feature across the neuron dimension so signals on very
        # different scales (a small top-down error vs a ~1.0 threshold) all reach
        # the encoders at unit scale; without this the error signal is swamped and
        # the rule cannot generalize past memorizing single batches.
        pre_feats = standardize(pre_feats, 0)
        post_feats = standardize(post_feats, 0)

        pre_embed = self.pre_encoder(pre_feats).view(pre, self.heads, self.head_dim)
        post_embed = self.post_encoder(post_feats).view(post, self.heads, self.head_dim)
        # Per-head bilinear score: [heads, post, pre]
        scores = torch.einsum("qhd,phd->hqp", post_embed, pre_embed) * self.score_scale

        channels = [weight, stats["potentiation"], stats["depression"], weight.abs()]
        if self.use_momentum:
            # The rule's own accumulated update trace -- the state that lets it act
            # like a momentum/adaptive optimizer rather than a memoryless map.
            channels.append(momentum if momentum is not None else torch.zeros_like(weight))
        syn = standardize(torch.stack(channels, dim=-1), (0, 1))   # [post, pre, syn_features]
        gated = self.gate(syn)                            # [post, pre, heads+1]
        head_gate = gated[..., :self.heads].permute(2, 0, 1)   # [heads, post, pre]
        bias = gated[..., self.heads]                     # [post, pre]
        delta = (head_gate * scores).sum(dim=0) + bias    # [post, pre]
        return delta

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def zero_error(device):
    """Two-factor mode: no top-down error, so the third factor is identically 0."""
    return [torch.zeros(HIDDEN_SIZE, device=device) for _ in range(HIDDEN_LAYERS)]


def zero_momentum(device):
    """Per-synapse momentum state, one [post, pre] trace per layer."""
    return [torch.zeros((HIDDEN_SIZE, fan_in_of(i)), device=device) for i in range(HIDDEN_LAYERS)]


# --------------------------------------------------------------------------- #
# Applying an update direction (identical constraints for every rule)         #
# --------------------------------------------------------------------------- #
def apply_update(weight, direction, lr, weight_norm):
    """W <- renorm(W + lr * rowdir(direction)). Same transform for policy and teacher."""
    return normalize_rows(weight + lr * normalize_rows(direction, 1.0), weight_norm)


# --------------------------------------------------------------------------- #
# Meta-training: gradient imitation along a teacher-forced surrogate-GD roll  #
# --------------------------------------------------------------------------- #
def surrogate_gradients(weights, thresholds, pixels, labels, readout_head, config):
    """Under the surrogate-spike forward, return per-layer dL/dW (the teacher
    target), per-neuron top-down error dL/dprobe (the third factor, consistent
    with that target), the readout-head gradients, and the loss."""
    with torch.enable_grad():   # robust to being called inside a @no_grad deploy
        leaves = [w.detach().clone().requires_grad_(True) for w in weights]
        probes = [torch.zeros(HIDDEN_SIZE, device=pixels.device, requires_grad=True) for _ in weights]
        features, _ = run_snn(leaves, thresholds, pixels, config, "surrogate", collect=False, probes=probes)
        logits = readout_head(torch.cat(features, dim=1))
        loss = F.cross_entropy(logits, labels)
        grads = torch.autograd.grad(loss, leaves + probes + list(readout_head.parameters()))
    weight_grads = grads[:HIDDEN_LAYERS]
    errors = grads[HIDDEN_LAYERS:2 * HIDDEN_LAYERS]
    head_grads = grads[2 * HIDDEN_LAYERS:]
    return weight_grads, errors, head_grads, loss.detach()


def row_cosine(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1).clamp_min(1e-8) * b.norm(dim=1).clamp_min(1e-8)
    return (num / den).mean()


def meta_train(policy, train, config, opt, device):
    train_x, train_y = train
    rollout_rng = random.Random(opt.seed0)
    head = nn.Linear(HIDDEN_LAYERS * HIDDEN_SIZE, CLASSES).to(device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=opt.meta_lr)
    head_opt = torch.optim.Adam(head.parameters(), lr=opt.head_lr)

    log_every = max(1, opt.meta_log_every)
    step = 0
    running_cos = 0.0
    running_layer_cos = [0.0] * HIDDEN_LAYERS
    running_loss = 0.0
    start = time.monotonic()
    history = []
    episode = 0
    while step < opt.meta_steps:
        # Fresh teacher-forced rollout from a new SNN init and shuffle.
        init_gen = torch.Generator(device=device).manual_seed(opt.seed0 * 7919 + episode)
        weights, thresholds = init_snn(device, init_gen, config.weight_norm, config.initial_threshold)
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(episode + 1))
        episode += 1
        for start_idx in range(0, len(order), opt.meta_batch):
            if step >= opt.meta_steps:
                break
            indices = order[start_idx:start_idx + opt.meta_batch]
            pixels = train_x[indices].float() / 255.0
            labels = train_y[indices]

            features, stats = run_snn(weights, thresholds, pixels, config, "hard", collect=True)

            weight_grads, true_errors, head_grads, loss = surrogate_gradients(
                weights, thresholds, pixels, labels, head, config)
            for param, grad in zip(head.parameters(), head_grads):
                param.grad = grad
            head_opt.step()
            head_opt.zero_grad(set_to_none=True)

            errors = true_errors if opt.factor == "three" else zero_error(device)

            policy_opt.zero_grad(set_to_none=True)
            cos_sum = 0.0
            policy_dirs = []
            for index in range(HIDDEN_LAYERS):
                layer_stats = {
                    "potentiation": stats[index]["potentiation"],
                    "depression": stats[index]["depression"],
                    "pre_rate": stats[index]["pre_rate"],
                    "post_rate": stats[index]["post_rate"],
                    "threshold": thresholds[index],
                }
                direction = policy(weights[index], layer_stats, errors[index])
                target = -weight_grads[index]
                cos = row_cosine(direction, target)
                (1.0 - cos).backward()
                cos_sum += cos.item()
                running_layer_cos[index] += cos.item()
                policy_dirs.append(direction.detach())
            policy_opt.step()

            # Advance the SNN. Teacher forcing (surrogate-GD) is stable but only
            # ever shows the policy the teacher's trajectory; the policy then
            # diverges at deployment. DAgger fixes this by advancing along the
            # POLICY's own direction so it learns to recover from the states it
            # will actually visit -- but rolling out an early, random policy
            # collapses the SNN and teaches garbage. So ramp the rollout
            # probability from 0 up to rollout_prob only after a warmup, once the
            # policy is competent. The imitation target stays the true gradient.
            warmup = opt.rollout_warmup * opt.meta_steps
            ramp = max(1.0, opt.meta_steps - warmup)
            prob = opt.rollout_prob * min(1.0, max(0.0, (step - warmup) / ramp))
            use_policy = rollout_rng.random() < prob
            with torch.no_grad():
                for index in range(HIDDEN_LAYERS):
                    step_dir = policy_dirs[index] if use_policy else -weight_grads[index]
                    weights[index] = apply_update(
                        weights[index], step_dir, opt.inner_lr, config.weight_norm)

            running_cos += cos_sum / HIDDEN_LAYERS
            running_loss += loss.item()
            step += 1
            if step % log_every == 0:
                mean_cos = running_cos / log_every
                mean_loss = running_loss / log_every
                per_layer = "/".join(f"{c / log_every:.2f}" for c in running_layer_cos)
                elapsed = time.monotonic() - start
                print(f"  meta step {step:6d}/{opt.meta_steps}  align cos {mean_cos:+.4f}  "
                      f"(L {per_layer})  teacher loss {mean_loss:.4f}  {elapsed:.0f}s", flush=True)
                history.append((step, mean_cos, mean_loss))
                running_cos = 0.0
                running_layer_cos = [0.0] * HIDDEN_LAYERS
                running_loss = 0.0
    return history


def policy_directions(policy, weights, thresholds, stats, errors, momentum=None, beta=0.0):
    """Per-layer update directions to APPLY, plus the carried momentum state.

    With momentum, the policy reads its accumulated update trace and the applied
    direction is the heavy-ball smoothing m <- beta*m + policy_output; without it
    the raw policy output is applied and momentum stays None."""
    directions = []
    for index in range(HIDDEN_LAYERS):
        layer_stats = {
            "potentiation": stats[index]["potentiation"],
            "depression": stats[index]["depression"],
            "pre_rate": stats[index]["pre_rate"],
            "post_rate": stats[index]["post_rate"],
            "threshold": thresholds[index],
        }
        mom = momentum[index] if momentum is not None else None
        directions.append(policy(weights[index], layer_stats, errors[index], momentum=mom))
    if momentum is None:
        return directions, None
    smoothed = [beta * momentum[i] + directions[i] for i in range(HIDDEN_LAYERS)]
    return smoothed, smoothed


def meta_train_unroll(policy, train, config, opt, device):
    """Learned-optimizer objective: unroll the policy's own updates for K steps
    and minimize the readout loss *after* the rollout, backpropagating through the
    updates. Unlike gradient imitation this penalizes exactly the trajectory-
    compounding error that sinks the imitated rule at deployment. The persistent
    weights walk the (stable) surrogate-GD teacher trajectory so the policy is
    trained to make progress from every state a good optimizer visits."""
    train_x, train_y = train
    rollout_rng = random.Random(opt.seed0)
    head = nn.Linear(HIDDEN_LAYERS * HIDDEN_SIZE, CLASSES).to(device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=opt.meta_lr)
    head_opt = torch.optim.Adam(head.parameters(), lr=opt.head_lr)
    K, lr, wn = opt.unroll, opt.inner_lr, config.weight_norm
    log_every = max(1, opt.meta_log_every)
    step, episode = 0, 0
    running_loss, running_gain = 0.0, 0.0
    start = time.monotonic()
    history = []

    def batch(order, at):
        idx = order[at:at + opt.meta_batch]
        return train_x[idx].float() / 255.0, train_y[idx]

    def current_horizon():
        # Curriculum: start with short (healthy) self-trajectories and grow to the
        # full horizon by 70% of training, so the policy is never asked to keep a
        # trajectory stable for longer than it can before it has learned to.
        frac = min(1.0, step / max(1.0, 0.7 * opt.meta_steps))
        return max(opt.horizon_start, int(opt.horizon_start + (opt.horizon - opt.horizon_start) * frac))

    window, weights, thresholds, horizon = 0, None, None, opt.horizon_start
    while step < opt.meta_steps:
        if weights is None or window >= horizon:
            # Reset to a fresh SNN every `horizon` windows; between resets the
            # trajectory continues (reshuffling data), so the policy is trained over
            # progressively longer trajectories rather than only short ones.
            init_gen = torch.Generator(device=device).manual_seed(opt.seed0 * 7919 + episode)
            weights, thresholds = init_snn(device, init_gen, wn, config.initial_threshold)
            momentum = zero_momentum(device) if opt.momentum > 0 else None
            episode += 1
            window = 0
            horizon = current_horizon()
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(episode * 100003 + window))
        base = 0
        while (step < opt.meta_steps and window < horizon
               and base + (K + 1) * opt.meta_batch <= len(order)):
            query_px, query_lb = batch(order, base + K * opt.meta_batch)

            # Differentiable K-step policy rollout branching off the current state.
            rolled = [w.detach() for w in weights]
            rolled_m = [m.detach() for m in momentum] if momentum is not None else None
            for j in range(K):
                px, lb = batch(order, base + j * opt.meta_batch)
                frozen = [w.detach() for w in rolled]
                with torch.no_grad():
                    _, stats = run_snn(frozen, thresholds, px, config, "hard", collect=True)
                if opt.factor == "three":
                    _, errors, _, _ = surrogate_gradients(frozen, thresholds, px, lb, head, config)
                else:
                    errors = zero_error(device)
                directions, rolled_m = policy_directions(
                    policy, rolled, thresholds, stats, errors, rolled_m, opt.momentum)
                rolled = [apply_update(rolled[i], directions[i], lr, wn) for i in range(HIDDEN_LAYERS)]

            features, _ = run_snn(rolled, thresholds, query_px, config, "surrogate", collect=False)
            loss = F.cross_entropy(head(torch.cat(features, dim=1)), query_lb)
            with torch.no_grad():
                base_feats, _ = run_snn([w.detach() for w in weights], thresholds, query_px,
                                        config, "surrogate", collect=False)
                base_loss = F.cross_entropy(head(torch.cat(base_feats, dim=1)), query_lb)

            policy_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)
            policy_opt.step()
            head_opt.step()

            # Advance the persistent state. Early on, walk the stable teacher
            # trajectory; once warmed up, ramp toward advancing with the policy's
            # OWN (detached) rollout -- reusing the branch just computed -- so it
            # trains on the state distribution it will visit at deployment. The
            # unroll objective makes this self-correcting where DAgger imitation
            # was not: every window is optimized to reduce loss from wherever it is.
            warmup = opt.rollout_warmup * opt.meta_steps
            ramp = max(1.0, opt.meta_steps - warmup)
            prob = opt.rollout_prob * min(1.0, max(0.0, (step - warmup) / ramp))
            if rollout_rng.random() < prob:
                weights = [w.detach() for w in rolled]
            else:
                with torch.no_grad():
                    for j in range(K):
                        px, lb = batch(order, base + j * opt.meta_batch)
                        grads, _, _, _ = surrogate_gradients(weights, thresholds, px, lb, head, config)
                        weights = [apply_update(weights[i], -grads[i], lr, wn) for i in range(HIDDEN_LAYERS)]
            base += (K + 1) * opt.meta_batch
            window += 1

            running_loss += loss.item()
            running_gain += (base_loss.item() - loss.item())
            step += 1
            if step % log_every == 0:
                elapsed = time.monotonic() - start
                print(f"  unroll step {step:6d}/{opt.meta_steps}  query loss {running_loss / log_every:.4f}  "
                      f"K-step gain {running_gain / log_every:+.4f}  {elapsed:.0f}s", flush=True)
                history.append((step, running_loss / log_every, running_gain / log_every))
                running_loss, running_gain = 0.0, 0.0
    return history


# --------------------------------------------------------------------------- #
# Deployment: train a fresh SNN with a fixed rule, then ridge-read it.        #
# --------------------------------------------------------------------------- #
class RidgeHead(nn.Module):
    """Frozen linear readout (from a ridge fit) used as a differentiable head so
    the true top-down error can be backpropagated at deployment. No parameters."""
    def __init__(self, mean, std, readout):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("readout", readout)

    def forward(self, concat):
        standardized = (concat - self.mean) / self.std
        design = torch.cat((standardized, torch.ones((len(concat), 1), device=concat.device)), dim=1)
        return design @ self.readout


@torch.no_grad()
def deploy(rule, policy, train, test, config, opt, device, seed):
    """rule in {'policy', 'teacher'}. Returns per-layer + concat test/train acc."""
    train_x, train_y = train
    init_gen = torch.Generator(device=device).manual_seed(seed)
    weights, thresholds = init_snn(device, init_gen, config.weight_norm, config.initial_threshold)
    needs_head = rule == "teacher" or opt.factor == "three"
    head = None
    lr = opt.deploy_lr if opt.deploy_lr > 0.0 else opt.inner_lr

    for epoch in range(opt.deploy_epochs):
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(seed * 101 + epoch))
        if needs_head:
            features = extract_features(weights, thresholds, train_x, config, opt.eval_batch)
            head = RidgeHead(*fit_ridge(torch.cat(features, dim=1), train_y, opt.ridge)).to(device)
        for start_idx in range(0, len(order), opt.deploy_batch):
            indices = order[start_idx:start_idx + opt.deploy_batch]
            pixels = train_x[indices].float() / 255.0
            labels = train_y[indices]
            if rule == "teacher":
                grads, _, _, _ = surrogate_gradients(weights, thresholds, pixels, labels, head, config)
                directions = [-g for g in grads]
            else:
                _, stats = run_snn(weights, thresholds, pixels, config, "hard", collect=True)
                if opt.factor == "three":
                    _, errors, _, _ = surrogate_gradients(weights, thresholds, pixels, labels, head, config)
                else:
                    errors = zero_error(device)
                directions = []
                for index in range(HIDDEN_LAYERS):
                    layer_stats = {
                        "potentiation": stats[index]["potentiation"],
                        "depression": stats[index]["depression"],
                        "pre_rate": stats[index]["pre_rate"],
                        "post_rate": stats[index]["post_rate"],
                        "threshold": thresholds[index],
                    }
                    directions.append(policy(weights[index], layer_stats, errors[index]))
            for index in range(HIDDEN_LAYERS):
                weights[index] = apply_update(weights[index], directions[index],
                                              lr, config.weight_norm)

    return evaluate(weights, thresholds, train, test, config, opt)


def corrupt_direction(gradient_dir, cos_target, generator):
    """A per-row unit vector with a fixed cosine to gradient_dir, the rest fresh
    independent noise. Models a gradient estimate of a given quality whose error
    is *unbiased* (re-randomized every step), unlike a deterministic policy."""
    if cos_target >= 0.999:
        return gradient_dir
    unit = gradient_dir / gradient_dir.norm(dim=1, keepdim=True).clamp_min(1e-8)
    noise = torch.randn(gradient_dir.shape, device=gradient_dir.device, generator=generator)
    noise = noise - (noise * unit).sum(dim=1, keepdim=True) * unit
    noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return cos_target * unit + math.sqrt(max(0.0, 1.0 - cos_target ** 2)) * noise


@torch.no_grad()
def deploy_synthetic(cos_target, train, test, config, opt, device, seed):
    """Train the SNN with synthetic directions of a controlled cosine to the true
    surrogate gradient. Isolates how much gradient-cosine a deployment needs,
    independent of any policy -- the control the learned rule is measured against."""
    train_x, train_y = train
    init_gen = torch.Generator(device=device).manual_seed(seed)
    weights, thresholds = init_snn(device, init_gen, config.weight_norm, config.initial_threshold)
    noise_gen = torch.Generator(device=device).manual_seed(seed * 7 + 1)
    lr = opt.deploy_lr if opt.deploy_lr > 0.0 else opt.inner_lr
    for epoch in range(opt.deploy_epochs):
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(seed * 101 + epoch))
        features = extract_features(weights, thresholds, train_x, config, opt.eval_batch)
        head = RidgeHead(*fit_ridge(torch.cat(features, dim=1), train_y, opt.ridge)).to(device)
        for start_idx in range(0, len(order), opt.deploy_batch):
            indices = order[start_idx:start_idx + opt.deploy_batch]
            pixels = train_x[indices].float() / 255.0
            labels = train_y[indices]
            grads, _, _, _ = surrogate_gradients(weights, thresholds, pixels, labels, head, config)
            for index in range(HIDDEN_LAYERS):
                weights[index] = apply_update(
                    weights[index], corrupt_direction(-grads[index], cos_target, noise_gen),
                    lr, config.weight_norm)
    return evaluate(weights, thresholds, train, test, config, opt)


@torch.no_grad()
def extract_features(weights, thresholds, images, config, batch_size):
    outputs = [torch.empty((len(images), HIDDEN_SIZE), device=images.device) for _ in weights]
    for start_idx in range(0, len(images), batch_size):
        chunk = images[start_idx:start_idx + batch_size].float() / 255.0
        features, _ = run_snn(weights, thresholds, chunk, config, "hard", collect=True)
        for index in range(HIDDEN_LAYERS):
            outputs[index][start_idx:start_idx + len(chunk)] = features[index]
    return outputs


@torch.no_grad()
def evaluate(weights, thresholds, train, test, config, opt):
    train_x, train_y = train
    test_x, test_y = test
    train_features = extract_features(weights, thresholds, train_x, config, opt.eval_batch)
    test_features = extract_features(weights, thresholds, test_x, config, opt.eval_batch)
    result = {"layers": []}
    for index in range(HIDDEN_LAYERS):
        mean, std, readout = fit_ridge(train_features[index], train_y, opt.ridge)
        _, train_acc = ridge_scores(train_features[index], train_y, mean, std, readout)
        _, test_acc = ridge_scores(test_features[index], test_y, mean, std, readout)
        result["layers"].append({"train_acc": train_acc, "test_acc": test_acc})
    mean, std, readout = fit_ridge(torch.cat(train_features, dim=1), train_y, opt.ridge)
    _, ctrain = ridge_scores(torch.cat(train_features, dim=1), train_y, mean, std, readout)
    _, ctest = ridge_scores(torch.cat(test_features, dim=1), test_y, mean, std, readout)
    result["concat"] = {"train_acc": ctrain, "test_acc": ctest}
    return result


def pixel_baseline(train, test, ridge):
    train_x, train_y = train
    test_x, test_y = test
    mean, std, readout = fit_ridge(train_x.float(), train_y, ridge)
    _, train_acc = ridge_scores(train_x.float(), train_y, mean, std, readout)
    _, test_acc = ridge_scores(test_x.float(), test_y, mean, std, readout)
    return {"train_acc": train_acc, "test_acc": test_acc}


def format_result(name, result):
    layers = "  ".join(f"L{i + 1} {100 * layer['test_acc']:.2f}%"
                       for i, layer in enumerate(result["layers"]))
    return f"  {name:20s} {layers}  concat {100 * result['concat']['test_acc']:.2f}%"


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
def self_test(device, opt):
    print(f"kmnist_metaplasticity self-test: device={device} "
          f"({torch.cuda.get_device_name(device)})")
    config = SNNConfig(timesteps=6)
    policy = PlasticityPolicy(width=opt.width, depth=opt.depth,
                              heads=opt.heads, head_dim=opt.head_dim).to(device)
    params = policy.parameter_count()
    print(f"  policy parameters: {params:,}")
    if params < 50_000_000:
        raise AssertionError(f"policy has {params:,} params, need >= 50M")

    gen = torch.Generator(device=device).manual_seed(0)
    weights, thresholds = init_snn(device, gen, config.weight_norm, config.initial_threshold)
    head = nn.Linear(HIDDEN_LAYERS * HIDDEN_SIZE, CLASSES).to(device)
    pixels = torch.rand((16, INPUT_SIZE), device=device, generator=gen)
    labels = torch.randint(0, CLASSES, (16,), device=device, generator=gen)

    # Validate the differentiable recurrence plumbing through all four layers with
    # a smooth spike primitive and all-winner masks (top_k = HIDDEN), so finite
    # differences are a valid check of the analytic gradient. Run it in float64:
    # the true loss changes here (~1e-7) sit below float32's resolution of a ~2.3
    # cross-entropy, which would make finite differences read spurious zeros.
    check = SNNConfig(timesteps=4, beta=config.beta, top_k=HIDDEN_SIZE,
                      trace_decay=config.trace_decay, alpha=config.alpha,
                      weight_norm=config.weight_norm, initial_threshold=config.initial_threshold)
    weights64 = [w.double() for w in weights]
    thresholds64 = [t.double() for t in thresholds]
    pixels64 = pixels.double()
    head64 = nn.Linear(HIDDEN_LAYERS * HIDDEN_SIZE, CLASSES).double().to(device)
    head64.load_state_dict({k: v.double() for k, v in head.state_dict().items()})

    def soft_loss(perturbed):
        features, _ = run_snn(perturbed, thresholds64, pixels64, check, "soft", collect=False)
        return F.cross_entropy(head64(torch.cat(features, dim=1)), labels)

    leaves = [w.detach().clone().requires_grad_(True) for w in weights64]
    soft_grads = torch.autograd.grad(soft_loss(leaves), leaves)
    eps = 1e-4
    checked, agree = 0, 0
    for layer in range(HIDDEN_LAYERS):
        for (i, j) in [(3, 5), (100, 20), (200, fan_in_of(layer) - 1)]:
            plus = [w.clone() for w in weights64]
            minus = [w.clone() for w in weights64]
            plus[layer][i, j] += eps
            minus[layer][i, j] -= eps
            with torch.no_grad():
                numeric = ((soft_loss(plus) - soft_loss(minus)) / (2 * eps)).item()
            analytic = soft_grads[layer][i, j].item()
            checked += 1
            if abs(numeric - analytic) <= 1e-8 + 1e-4 * abs(analytic):
                agree += 1
    if agree < checked:
        raise AssertionError(f"differentiable forward disagreed with finite differences ({agree}/{checked})")
    print(f"  differentiable forward vs finite differences: {agree}/{checked} agree")

    # Surrogate-mode gradients (hard forward) must be finite for the real config.
    grads, _, _, _ = surrogate_gradients(weights, thresholds, pixels, labels, head, config)
    if any(not torch.isfinite(g).all() for g in grads):
        raise AssertionError("surrogate gradient is not finite")
    print("  surrogate gradient finite across all layers: pass")

    # A single meta step must run end to end and produce a finite cosine.
    errors = [torch.zeros(HIDDEN_SIZE, device=device) for _ in range(HIDDEN_LAYERS)]
    _, stats = run_snn(weights, thresholds, pixels, config, "hard", collect=True)
    layer_stats = {**{k: stats[0][k] for k in ("potentiation", "depression", "pre_rate", "post_rate")},
                   "threshold": thresholds[0]}
    direction = policy(weights[0], layer_stats, errors[0])
    if direction.shape != weights[0].shape or not torch.isfinite(direction).all():
        raise AssertionError("policy produced a bad update tensor")
    cos = row_cosine(direction, -grads[0])
    if not math.isfinite(cos.item()):
        raise AssertionError("alignment cosine is not finite")
    print(f"  policy forward + alignment: pass (init cos {cos.item():+.3f})")
    print("  all self-test checks passed")


# --------------------------------------------------------------------------- #
# Subnetwork optimizer: BPTT localizes the responsible subnetwork per sample,  #
# a learned policy makes the focused edit, trained on the loss after the edit. #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_sample_rates(weights, thresholds, pixels, config):
    """Per-sample post-neuron firing rate [B, HIDDEN] for each layer (hard forward)."""
    batch = pixels.shape[0]
    device = pixels.device
    dtype = weights[0].dtype
    voltages = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]
    previous = [torch.zeros_like(voltages[0]) for _ in weights]
    rates = [torch.zeros((batch, HIDDEN_SIZE), device=device, dtype=dtype) for _ in weights]
    for _ in range(config.timesteps):
        signal = pixels
        for index, weight in enumerate(weights):
            voltage = voltages[index] * config.beta + signal @ weight.t()
            voltage = voltage - previous[index] * thresholds[index]
            score = voltage - thresholds[index]
            spike = winner_mask(score, config.top_k) * (score >= 0).to(dtype)
            voltages[index] = voltage
            previous[index] = spike
            rates[index] = rates[index] + spike
            signal = spike
    return [rate / config.timesteps for rate in rates]


def sample_credit(weights, thresholds, pixels, labels, head, config):
    """Per-sample per-neuron error e[b,i] = dL/dprobe -- the BPTT signal for which
    neurons were to blame for THIS sample -- in a single backward pass."""
    with torch.enable_grad():
        leaves = [w.detach() for w in weights]
        probes = [torch.zeros((pixels.shape[0], HIDDEN_SIZE), device=pixels.device, requires_grad=True)
                  for _ in weights]
        features, _ = run_snn(leaves, thresholds, pixels, config, "surrogate", collect=False, probes=probes)
        loss = F.cross_entropy(head(torch.cat(features, dim=1)), labels)
        errors = torch.autograd.grad(loss, probes)
    return [error.detach() for error in errors], loss.detach()


class SubnetworkPolicy(nn.Module):
    """BPTT says WHICH post-neurons are to blame for each sample (top |e[b,i]|);
    this policy says HOW to edit their incoming weights. Per-(sample, neuron)
    encoders emit rank-H correction factors whose gated outer product, summed over
    the batch, is the weight update -- the same structure as the gradient
    (sum_b e_b (x) x_b), but with learned factors focused on the responsible
    subnetwork. The per-sample factoring recovers the sample covariance that
    batch-aggregation (the dense rule's cosine-0.3 ceiling) threw away."""

    POST_FEATURES = 4    # [error, |error|, post_rate, threshold]
    PRE_FEATURES = 3     # [pre_rate, sqrt, square]

    def __init__(self, width=2560, depth=3, heads=16, select_frac=0.25):
        super().__init__()
        self.heads = heads
        self.select_frac = select_frac
        self.post_encoder = mlp(self.POST_FEATURES, width, depth, heads)
        self.pre_encoder = mlp(self.PRE_FEATURES, width, depth, heads)

    def forward(self, weight, error, post_rate, threshold, pre_rate):
        batch, post = error.shape
        thr = threshold.unsqueeze(0).expand(batch, -1)
        post_feat = standardize(torch.stack([error, error.abs(), post_rate, thr], dim=-1), (0, 1))
        pre_feat = standardize(torch.stack([pre_rate, pre_rate.sqrt(), pre_rate ** 2], dim=-1), (0, 1))
        c_post = self.post_encoder(post_feat)               # [B, post, H]
        c_pre = self.pre_encoder(pre_feat)                  # [B, pre, H]
        # Responsible subnetwork: the top-fraction of post neurons per sample by
        # |error| (BPTT credit). A detached hard mask -- the selection is BPTT's
        # job, the edit is the policy's.
        keep = max(1, int(self.select_frac * post))
        index = error.abs().topk(keep, dim=1).indices
        gate = torch.zeros_like(error).scatter_(1, index, 1.0)
        gated = gate.unsqueeze(-1) * c_post                 # [B, post, H]
        return torch.einsum("bph,bqh->pq", gated, c_pre)    # [post, pre]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class AttentionBlock(nn.Module):
    """Pre-norm transformer block: the responsible neurons talk to each other."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class CoordinatedSubnetworkPolicy(nn.Module):
    """The subnetwork policy, but the responsible neurons COORDINATE.

    BPTT still selects which post-neurons are to blame for each sample. Those
    neurons are then passed through self-attention blocks so each one's edit is
    decided *in the context of what the others are doing* -- the coordinated,
    multi-weight intervention an expert would make, which independent per-neuron
    factors (the pointwise policy) structurally cannot express. Each coordinated
    neuron then queries the presynaptic population to decide which of its incoming
    weights to move. Capacity lives in the coordination, not in pointwise width."""

    def __init__(self, dim=1024, blocks=4, heads=8, select_frac=0.25):
        super().__init__()
        self.select_frac = select_frac
        self.heads = heads
        self.post_in = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pre_in = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.coordinate = nn.ModuleList([AttentionBlock(dim, heads) for _ in range(blocks)])
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.scale = 1.0 / math.sqrt(dim // heads)

    def forward(self, weight, error, post_rate, threshold, pre_rate):
        batch, post = error.shape
        pre = pre_rate.shape[1]
        thr = threshold.unsqueeze(0).expand(batch, -1)
        post_feat = standardize(torch.stack([error, error.abs(), post_rate, thr], dim=-1), (0, 1))
        pre_feat = standardize(torch.stack([pre_rate, pre_rate.sqrt(), pre_rate ** 2], dim=-1), (0, 1))

        # BPTT picks the responsible subnetwork for each sample.
        keep = max(1, int(self.select_frac * post))
        index = error.abs().topk(keep, dim=1).indices                      # [B, keep]
        chosen = torch.gather(post_feat, 1, index.unsqueeze(-1).expand(-1, -1, post_feat.shape[-1]))

        hidden = self.post_in(chosen)                                       # [B, keep, dim]
        for block in self.coordinate:
            hidden = block(hidden)                                          # neurons coordinate

        keys = self.pre_in(pre_feat)                                        # [B, pre, dim]
        query = self.to_query(hidden).view(batch, keep, self.heads, -1)
        key = self.to_key(keys).view(batch, pre, self.heads, -1)
        edit = torch.einsum("bkhd,bphd->bkp", query, key) * self.scale      # [B, keep, pre]

        # Accumulate each sample's subnetwork edit into the shared weight update.
        flat = index.reshape(-1)
        return torch.zeros_like(weight).index_add(0, flat, edit.reshape(batch * keep, pre))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def subnetwork_deltas(policy, weights, thresholds, errors, rates, pixels):
    """Per-layer weight edits from per-sample credit and the subnetwork policy."""
    deltas = []
    for index in range(HIDDEN_LAYERS):
        pre_rate = pixels if index == 0 else rates[index - 1]
        deltas.append(policy(weights[index], errors[index], rates[index], thresholds[index], pre_rate))
    return deltas


def meta_train_subnetwork(policy, train, config, opt, device):
    """Single-step lookahead: apply the focused edit, minimize the readout loss
    on a query batch after it, backprop into the policy. The persistent weights
    walk their own (curriculum-controlled) trajectory, as in the unroll trainer."""
    train_x, train_y = train
    rollout_rng = random.Random(opt.seed0)
    head = nn.Linear(HIDDEN_LAYERS * HIDDEN_SIZE, CLASSES).to(device)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=opt.meta_lr)
    head_opt = torch.optim.Adam(head.parameters(), lr=opt.head_lr)
    lr, wn = opt.inner_lr, config.weight_norm
    log_every = max(1, opt.meta_log_every)
    step, episode, window = 0, 0, 0
    weights, thresholds, horizon = None, None, opt.horizon_start
    running_loss, running_gain = 0.0, 0.0
    start = time.monotonic()
    history = []

    def batch(order, at):
        idx = order[at:at + opt.meta_batch]
        return train_x[idx].float() / 255.0, train_y[idx]

    def current_horizon():
        frac = min(1.0, step / max(1.0, 0.7 * opt.meta_steps))
        return max(opt.horizon_start, int(opt.horizon_start + (opt.horizon - opt.horizon_start) * frac))

    while step < opt.meta_steps:
        if weights is None or window >= horizon:
            init_gen = torch.Generator(device=device).manual_seed(opt.seed0 * 7919 + episode)
            weights, thresholds = init_snn(device, init_gen, wn, config.initial_threshold)
            episode += 1
            window = 0
            horizon = current_horizon()
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(episode * 100003 + window))
        base = 0
        while (step < opt.meta_steps and window < horizon
               and base + 2 * opt.meta_batch <= len(order)):
            px, lb = batch(order, base)
            query_px, query_lb = batch(order, base + opt.meta_batch)

            errors, _ = sample_credit(weights, thresholds, px, lb, head, config)
            rates = forward_sample_rates(weights, thresholds, px, config)
            deltas = subnetwork_deltas(policy, weights, thresholds, errors, rates, px)
            edited = [apply_update(weights[i], deltas[i], lr, wn) for i in range(HIDDEN_LAYERS)]

            features, _ = run_snn(edited, thresholds, query_px, config, "surrogate", collect=False)
            loss = F.cross_entropy(head(torch.cat(features, dim=1)), query_lb)
            with torch.no_grad():
                base_feats, _ = run_snn([w.detach() for w in weights], thresholds, query_px,
                                        config, "surrogate", collect=False)
                base_loss = F.cross_entropy(head(torch.cat(base_feats, dim=1)), query_lb)

            policy_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), opt.grad_clip)
            policy_opt.step()
            head_opt.step()

            warmup = opt.rollout_warmup * opt.meta_steps
            ramp = max(1.0, opt.meta_steps - warmup)
            prob = opt.rollout_prob * min(1.0, max(0.0, (step - warmup) / ramp))
            if rollout_rng.random() < prob:
                weights = [w.detach() for w in edited]
            else:
                with torch.no_grad():
                    grads, _, _, _ = surrogate_gradients(weights, thresholds, px, lb, head, config)
                    weights = [apply_update(weights[i], -grads[i], lr, wn) for i in range(HIDDEN_LAYERS)]
            base += 2 * opt.meta_batch
            window += 1

            running_loss += loss.item()
            running_gain += (base_loss.item() - loss.item())
            step += 1
            if step % log_every == 0:
                elapsed = time.monotonic() - start
                print(f"  subnet step {step:6d}/{opt.meta_steps}  query loss {running_loss / log_every:.4f}  "
                      f"edit gain {running_gain / log_every:+.4f}  {elapsed:.0f}s", flush=True)
                history.append((step, running_loss / log_every, running_gain / log_every))
                running_loss, running_gain = 0.0, 0.0
                # Checkpoint as we go: an interrupted run should not cost the whole
                # training, and the last checkpoint is still a deployable policy.
                if opt.save_policy:
                    torch.save({"state": policy.state_dict(), "opt": vars(opt), "step": step},
                               opt.save_policy)
    return history


@torch.no_grad()
def deploy_subnetwork(policy, train, test, config, opt, device, seed):
    train_x, train_y = train
    init_gen = torch.Generator(device=device).manual_seed(seed)
    weights, thresholds = init_snn(device, init_gen, config.weight_norm, config.initial_threshold)
    lr = opt.deploy_lr if opt.deploy_lr > 0.0 else opt.inner_lr
    for epoch in range(opt.deploy_epochs):
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(seed * 101 + epoch))
        features = extract_features(weights, thresholds, train_x, config, opt.eval_batch)
        head = RidgeHead(*fit_ridge(torch.cat(features, dim=1), train_y, opt.ridge)).to(device)
        for start_idx in range(0, len(order), opt.deploy_batch):
            indices = order[start_idx:start_idx + opt.deploy_batch]
            px = train_x[indices].float() / 255.0
            lb = train_y[indices]
            errors, _ = sample_credit(weights, thresholds, px, lb, head, config)
            rates = forward_sample_rates(weights, thresholds, px, config)
            deltas = subnetwork_deltas(policy, weights, thresholds, errors, rates, px)
            weights = [apply_update(weights[i], deltas[i], lr, config.weight_norm)
                       for i in range(HIDDEN_LAYERS)]
    return evaluate(weights, thresholds, train, test, config, opt)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run(opt):
    if not torch.cuda.is_available():
        fail("CUDA is required; no CUDA device is available")
    device = torch.device(opt.device)
    if device.type != "cuda":
        fail("this tool is CUDA-only; --device must name a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(opt.seed0)

    config = SNNConfig(timesteps=opt.timesteps, beta=opt.beta, top_k=opt.top_k,
                       trace_decay=opt.trace_decay, alpha=opt.alpha,
                       weight_norm=opt.weight_norm, initial_threshold=opt.threshold)
    if opt.subnetwork and opt.coordinate:
        policy = CoordinatedSubnetworkPolicy(dim=opt.coord_dim, blocks=opt.coord_blocks,
                                             heads=opt.coord_heads,
                                             select_frac=opt.select_frac).to(device)
    elif opt.subnetwork:
        policy = SubnetworkPolicy(width=opt.width, depth=opt.depth,
                                  heads=opt.heads, select_frac=opt.select_frac).to(device)
    else:
        policy = PlasticityPolicy(width=opt.width, depth=opt.depth, heads=opt.heads,
                                  head_dim=opt.head_dim, use_momentum=(opt.momentum > 0)).to(device)
    kind = ("subnetwork+coordination" if opt.subnetwork and opt.coordinate
            else "subnetwork" if opt.subnetwork else "dense")
    print(f"policy parameters: {policy.parameter_count():,}  factor={opt.factor}  rule={kind}")

    train, test = load_dataset(opt.data, device, opt.train_limit, opt.test_limit)
    print(f"kmnist_metaplasticity: {len(train[0])} train, {len(test[0])} test, "
          f"device={device} ({torch.cuda.get_device_name(device)}), torch={torch.__version__}")

    if opt.cosine_curve:
        cosines = [float(x) for x in opt.cosine_curve.split(",")]
        pixels = pixel_baseline(train, test, opt.ridge)
        print(f"\nsynthetic gradient-cosine curve ({opt.deploy_epochs} deploy epochs, seed {opt.seed0}):")
        print(f"  raw pixels control: concat {100 * pixels['test_acc']:.2f}%")
        for cos_target in cosines:
            result = deploy_synthetic(cos_target, train, test, config, opt, device, opt.seed0)
            print(format_result(f"cos={cos_target:.2f}", result), flush=True)
        return

    if opt.load_policy:
        policy.load_state_dict(torch.load(opt.load_policy, map_location=device)["state"])
        policy.eval()
        print(f"loaded policy from {opt.load_policy} (skipping meta-training)", flush=True)
    else:
        objective = ("subnetwork edit" if opt.subnetwork else
                     "differentiable unroll" if opt.unroll > 0 else "gradient imitation")
        print(f"meta-training the plasticity policy ({objective})...", flush=True)
        meta_start = time.monotonic()
        if opt.subnetwork:
            history = meta_train_subnetwork(policy, train, config, opt, device)
        elif opt.unroll > 0:
            history = meta_train_unroll(policy, train, config, opt, device)
        else:
            history = meta_train(policy, train, config, opt, device)
        meta_seconds = time.monotonic() - meta_start
        print(f"meta-training done in {meta_seconds:.0f}s", flush=True)
        if opt.save_policy:
            torch.save({"state": policy.state_dict(), "opt": vars(opt)}, opt.save_policy)
            print(f"saved policy to {opt.save_policy}")

    pixels = pixel_baseline(train, test, opt.ridge)
    print(f"\nbenchmark (KMNIST test accuracy, {opt.seeds} seeds, "
          f"{opt.deploy_epochs} deploy epochs):")
    print(f"  {'raw pixels':20s} concat {100 * pixels['test_acc']:.2f}%")

    rows = []
    for repeat in range(opt.seeds):
        seed = opt.seed0 + repeat
        if opt.subnetwork:
            learned = deploy_subnetwork(policy, train, test, config, opt, device, seed)
        else:
            learned = deploy("policy", policy, train, test, config, opt, device, seed)
        print(format_result(f"learned/{opt.factor} s{seed}", learned), flush=True)
        row = {"tag": opt.tag, "seed": seed, "factor": opt.factor,
               "policy_params": policy.parameter_count(),
               "meta_steps": opt.meta_steps, "deploy_epochs": opt.deploy_epochs,
               "pixel_test_acc": pixels["test_acc"],
               "learned_concat_test_acc": learned["concat"]["test_acc"],
               "learned_concat_train_acc": learned["concat"]["train_acc"]}
        for index in range(HIDDEN_LAYERS):
            row[f"learned_layer{index + 1}_test_acc"] = learned["layers"][index]["test_acc"]
        if opt.include_teacher:
            teacher = deploy("teacher", policy, train, test, config, opt, device, seed)
            print(format_result(f"teacher s{seed}", teacher), flush=True)
            row["teacher_concat_test_acc"] = teacher["concat"]["test_acc"]
            for index in range(HIDDEN_LAYERS):
                row[f"teacher_layer{index + 1}_test_acc"] = teacher["layers"][index]["test_acc"]
        rows.append(row)

    if opt.csv and rows:
        fresh = not os.path.exists(opt.csv) or os.path.getsize(opt.csv) == 0
        with open(opt.csv, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            if fresh:
                writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {opt.csv}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/kmnist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--factor", choices=["two", "three"], default="three")
    # SNN
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--trace-decay", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--weight-norm", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    # policy size
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=48)
    # meta-training
    parser.add_argument("--meta-steps", type=int, default=8000)
    parser.add_argument("--meta-batch", type=int, default=128)
    parser.add_argument("--meta-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=3e-3)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--unroll", type=int, default=0,
                        help="if >0, train by differentiable K-step rollout (learned optimizer) "
                             "instead of gradient imitation")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="meta-gradient norm clip for the unroll objective")
    parser.add_argument("--horizon", type=int, default=350,
                        help="max windows before the unroll trajectory resets to a fresh SNN")
    parser.add_argument("--horizon-start", type=int, default=20,
                        help="initial horizon for the curriculum (grows to --horizon)")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="if >0, give the policy a per-synapse momentum state (heavy-ball beta)")
    parser.add_argument("--subnetwork", action="store_true",
                        help="per-sample credit -> responsible-subnetwork -> focused learned edit")
    parser.add_argument("--select-frac", type=float, default=0.25,
                        help="fraction of post-neurons kept as the responsible subnetwork per sample")
    parser.add_argument("--coordinate", action="store_true",
                        help="responsible neurons coordinate via self-attention before editing")
    parser.add_argument("--coord-dim", type=int, default=1024)
    parser.add_argument("--coord-blocks", type=int, default=4)
    parser.add_argument("--coord-heads", type=int, default=8)
    parser.add_argument("--rollout-prob", type=float, default=0.0,
                        help="peak fraction of inner steps advanced by the policy (DAgger) instead of the teacher")
    parser.add_argument("--rollout-warmup", type=float, default=0.3,
                        help="fraction of meta-steps to teacher-force before ramping in policy rollouts")
    parser.add_argument("--meta-log-every", type=int, default=100)
    # deployment / benchmark
    parser.add_argument("--deploy-epochs", type=int, default=3)
    parser.add_argument("--deploy-lr", type=float, default=0.0,
                        help="inner lr at deployment (0 = use --inner-lr)")
    parser.add_argument("--deploy-batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed0", type=int, default=1)
    parser.add_argument("--include-teacher", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--csv")
    parser.add_argument("--tag", default="meta")
    parser.add_argument("--save-policy")
    parser.add_argument("--load-policy", help="skip meta-training and deploy this saved policy")
    parser.add_argument("--cosine-curve",
                        help="comma-separated cosines; train the SNN with synthetic directions "
                             "of each cosine to the true gradient instead of a policy")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        fail("CUDA is required; no CUDA device is available")
    device = torch.device(opt.device)
    if device.type != "cuda":
        fail("this tool is CUDA-only; --device must name a CUDA device")
    torch.cuda.set_device(device)
    if opt.self_test:
        self_test(device, opt)
        return
    run(opt)


if __name__ == "__main__":
    main()
