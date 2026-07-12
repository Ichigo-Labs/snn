#!/usr/bin/env python3
"""Verified architecture editing: diagnose the SNN, propose a structural edit,
keep it only if it measurably helps on held-out data.

Every plasticity rule in tools/kmnist_metaplasticity.py makes thousands of small
*blind* weight nudges -- it never checks whether an edit helped -- so error
compounds and the trajectory collapses (the learned rules there land at 13-68%
against STDP's 73.6%). This tool inverts that. It makes a handful of large,
*discrete*, *verified* edits: apply -> measure on a held-out split -> keep only
on improvement, otherwise revert. The search is therefore monotone by
construction: it cannot damage the network, no matter how bad a proposal is.

It also does things gradient descent structurally cannot. Backprop can only
reweight existing connections; it cannot notice that no unit in the network
distinguishes 3 from 5 and *build* one. The edits here are the moves a human
expert makes: find the capacity the readout is wasting, find the distinction the
network is failing to draw, and repurpose the former to make the latter.

Data hygiene: edits are selected on a validation split carved out of TRAIN. The
test set is untouched until the final report.
"""

import argparse
import gzip
import math
import os
import struct
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 4
CLASSES = 10


def fail(message):
    raise SystemExit(f"kmnist_architect: {message}")


# --------------------------------------------------------------------------- #
# Data                                                                          #
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
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(count, stride)


def load_split(data_dir, device):
    train_x = read_idx(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), 0x803).to(device)
    train_y = read_idx(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), 0x801).view(-1).to(device).long()
    test_x = read_idx(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), 0x803).to(device)
    test_y = read_idx(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), 0x801).view(-1).to(device).long()
    return (train_x, train_y), (test_x, test_y)


# --------------------------------------------------------------------------- #
# The SNN -- identical to tools/kmnist_stdp.py so the comparison is apples to  #
# apples: same architecture, same drive, same weight constraint, same readout. #
# --------------------------------------------------------------------------- #
class Config:
    def __init__(self, timesteps=20, beta=0.95, top_k=32, weight_norm=2.0, threshold=1.0):
        self.timesteps = timesteps
        self.beta = beta
        self.top_k = top_k
        self.weight_norm = weight_norm
        self.threshold = threshold


def fan_in_of(index):
    return INPUT_SIZE if index == 0 else HIDDEN_SIZE


def normalize_rows(matrix, norm):
    matrix = matrix - matrix.mean(dim=-1, keepdim=True)
    return matrix * (norm / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def init_snn(device, generator, config):
    weights, thresholds = [], []
    for index in range(HIDDEN_LAYERS):
        fan_in = fan_in_of(index)
        basis = torch.randn((fan_in, HIDDEN_SIZE), device=device, generator=generator)
        orthogonal, triangular = torch.linalg.qr(basis, mode="reduced")
        sign = triangular.diagonal().sign().masked_fill_(triangular.diagonal().eq(0), 1.0)
        weights.append((orthogonal * sign).t().contiguous().mul_(config.weight_norm))
        thresholds.append(torch.full((HIDDEN_SIZE,), config.threshold, device=device))
    return weights, thresholds


@torch.no_grad()
def forward(weights, thresholds, images, config, batch_size=1024):
    """Return per-layer readout features [N, HIDDEN] and per-layer input rates.

    features[l] is the low-pass-filtered spike statistic the ridge readout sees
    (the same statistic tools/kmnist_stdp.py uses). inputs[l] is the mean
    presynaptic activity into layer l -- what a repurposed neuron must respond to.
    """
    total = len(images)
    features = [torch.empty((total, HIDDEN_SIZE), device=images.device) for _ in weights]
    inputs = [torch.empty((total, fan_in_of(i)), device=images.device) for i in range(HIDDEN_LAYERS)]
    for start in range(0, total, batch_size):
        pixels = images[start:start + batch_size].float() / 255.0
        count = len(pixels)
        voltage = [torch.zeros((count, HIDDEN_SIZE), device=pixels.device) for _ in weights]
        previous = [torch.zeros_like(voltage[0]) for _ in weights]
        filtered = [torch.zeros_like(voltage[0]) for _ in weights]
        summed = [torch.zeros_like(voltage[0]) for _ in weights]
        rates = [torch.zeros_like(voltage[0]) for _ in weights]
        for _ in range(config.timesteps):
            signal = pixels
            for index, weight in enumerate(weights):
                drive = voltage[index] * config.beta + signal @ weight.t()
                membrane = drive - previous[index] * thresholds[index]
                score = membrane - thresholds[index]
                _, winners = score.topk(config.top_k, dim=1, sorted=False)
                mask = torch.zeros_like(score).scatter_(1, winners, 1.0)
                spike = mask * (score >= 0).float()
                voltage[index] = membrane
                previous[index] = spike
                filtered[index] = filtered[index] * config.beta + spike
                summed[index] = summed[index] + filtered[index]
                rates[index] = rates[index] + spike
                signal = spike
        for index in range(HIDDEN_LAYERS):
            features[index][start:start + count] = summed[index] / config.timesteps
            inputs[index][start:start + count] = (
                pixels if index == 0 else rates[index - 1] / config.timesteps)
    return features, inputs


# --------------------------------------------------------------------------- #
# STDP pre-training (same rule and hyper-parameters as tools/kmnist_stdp.py).  #
# Verified editing is MONOTONE, so the sharp question is not whether it can    #
# out-run STDP from a random start -- it cannot, it only rebuilds a few dozen  #
# of 1024 units -- but whether it can REPAIR a trained net: fix the dead and   #
# redundant units, and draw the distinctions STDP never learned.               #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def train_stdp(weights, thresholds, train_x, config, opt, device):
    trace_decay, ltd_ratio = 0.5, 0.5
    target_rate, threshold_lr = 0.08, 0.05
    scale = min(1.0, 5000 / len(train_x))
    learning_rate = opt.stdp_lr * scale
    for epoch in range(opt.stdp_epochs):
        order = torch.randperm(len(train_x), device=device,
                               generator=torch.Generator(device=device).manual_seed(
                                   opt.seed0 * 100003 + epoch))
        for start in range(0, len(order), opt.stdp_batch):
            pixels = train_x[order[start:start + opt.stdp_batch]].float() / 255.0
            count = len(pixels)
            voltage = [torch.zeros((count, HIDDEN_SIZE), device=device) for _ in weights]
            previous = [torch.zeros_like(voltage[0]) for _ in weights]
            pre_trace = [torch.zeros((count, fan_in_of(i)), device=device) for i in range(HIDDEN_LAYERS)]
            post_trace = [torch.zeros_like(voltage[0]) for _ in weights]
            potentiation = [torch.zeros_like(w) for w in weights]
            depression = [torch.zeros_like(w) for w in weights]
            seen = [torch.zeros(HIDDEN_SIZE, device=device) for _ in weights]
            for _ in range(config.timesteps):
                signal = pixels
                for index, weight in enumerate(weights):
                    pre = signal
                    drive = voltage[index] * config.beta + pre @ weight.t()
                    membrane = drive - previous[index] * thresholds[index]
                    scores = membrane - thresholds[index]
                    _, winners = scores.topk(config.top_k, dim=1, sorted=False)
                    mask = torch.zeros_like(scores).scatter_(1, winners, 1.0)
                    spike = mask * (scores >= 0).float()
                    voltage[index] = membrane
                    pre_trace[index].mul_(trace_decay).add_(pre)
                    potentiation[index].addmm_(spike.t(), pre_trace[index])
                    depression[index].addmm_(post_trace[index].t(), pre)
                    post_trace[index].mul_(trace_decay).add_(spike)
                    previous[index] = spike
                    seen[index].add_(spike.sum(dim=0))
                    signal = spike
            for index in range(HIDDEN_LAYERS):
                eligibility = potentiation[index].add(depression[index], alpha=-ltd_ratio)
                eligibility.div_(seen[index].unsqueeze(1).clamp_min_(1.0))
                eligibility.sub_(eligibility.mean(dim=1, keepdim=True))
                eligibility.div_(eligibility.norm(dim=1, keepdim=True).clamp_min_(1e-8))
                weights[index].add_(eligibility, alpha=learning_rate)
                weights[index].copy_(normalize_rows(weights[index], config.weight_norm))
                rates = seen[index] / (count * config.timesteps)
                thresholds[index].add_(rates - target_rate, alpha=threshold_lr * scale)
                thresholds[index].clamp_(0.4, 2.0)
    return weights, thresholds


# --------------------------------------------------------------------------- #
# Readout (same closed-form ridge as the STDP benchmark)                       #
# --------------------------------------------------------------------------- #
def fit_ridge(features, labels, ridge):
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(0.02)
    design = torch.cat(((features - mean) / std,
                        torch.ones((len(features), 1), device=features.device)), dim=1)
    gram = design.t() @ design
    gram.diagonal().add_(ridge)
    readout = torch.linalg.solve(gram, design.t() @ F.one_hot(labels, CLASSES).float())
    return mean, std, readout


def apply_readout(features, mean, std, readout):
    design = torch.cat(((features - mean) / std,
                        torch.ones((len(features), 1), device=features.device)), dim=1)
    return design @ readout


def accuracy_of(logits, labels):
    return logits.argmax(dim=1).eq(labels).float().mean().item()


# --------------------------------------------------------------------------- #
# Score a candidate network: fit the readout on `fit`, score it on `val`.      #
# The concatenation of all four layers is the feature, as in kmnist_stdp.md.   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score(weights, thresholds, fit, val, config, ridge):
    """Return (val cross-entropy LOSS, val accuracy, readout).

    Selection uses the LOSS, not the accuracy. Repurposing one unit out of 1024
    shifts true accuracy by ~0.06% while the standard error of an accuracy
    estimate on a few thousand samples is ~0.8% -- the signal is an order of
    magnitude below the noise, so ranking edits by accuracy ranks noise. The
    cross-entropy is continuous: every sample's logits move, so it resolves small
    edits that accuracy cannot see at all.
    """
    fit_x, fit_y = fit
    val_x, val_y = val
    fit_features, _ = forward(weights, thresholds, fit_x, config)
    val_features, _ = forward(weights, thresholds, val_x, config)
    mean, std, readout = fit_ridge(torch.cat(fit_features, dim=1), fit_y, ridge)
    logits = apply_readout(torch.cat(val_features, dim=1), mean, std, readout)
    loss = F.cross_entropy(logits, val_y).item()
    return loss, accuracy_of(logits, val_y), readout


# --------------------------------------------------------------------------- #
# Diagnosis: what a human expert would look at before touching anything.       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diagnose(weights, thresholds, fit, val, config, ridge):
    """Return (val_accuracy, confused class pairs, wasted neurons, class-mean inputs).

    - wasted neurons: the readout's coefficient norm for each hidden unit. A unit
      the readout barely uses is capacity the network is throwing away -- exactly
      what an expert repurposes.
    - confused pairs: off-diagonal mass of the confusion matrix -- the
      distinctions the network is failing to draw.
    """
    fit_x, fit_y = fit
    val_x, val_y = val
    fit_features, fit_inputs = forward(weights, thresholds, fit_x, config)
    val_features, _ = forward(weights, thresholds, val_x, config)
    mean, std, readout = fit_ridge(torch.cat(fit_features, dim=1), fit_y, ridge)
    logits = apply_readout(torch.cat(val_features, dim=1), mean, std, readout)
    accuracy = accuracy_of(logits, val_y)
    del val_features

    # How much the readout actually leans on each hidden unit (drop the bias row).
    usage = readout[:-1].norm(dim=1).view(HIDDEN_LAYERS, HIDDEN_SIZE)   # [layer, neuron]

    predicted = logits.argmax(dim=1)
    confusion = torch.zeros((CLASSES, CLASSES), device=val_x.device)
    confusion.index_put_((val_y, predicted), torch.ones_like(val_y, dtype=torch.float), accumulate=True)
    confusion.fill_diagonal_(0.0)
    # Symmetrise: a<->b confusion is one distinction to fix.
    pair_cost = confusion + confusion.t()

    # Class-conditional mean input to each layer -- the template a repurposed
    # neuron is built from.
    class_inputs = []
    for index in range(HIDDEN_LAYERS):
        means = torch.zeros((CLASSES, fan_in_of(index)), device=val_x.device)
        for cls in range(CLASSES):
            selected = fit_inputs[index][fit_y == cls]
            if len(selected):
                means[cls] = selected.mean(dim=0)
        class_inputs.append(means)
    return accuracy, pair_cost, usage, class_inputs


def top_confused_pairs(pair_cost, count):
    flat = torch.triu(pair_cost, diagonal=1).flatten()
    values, indices = flat.topk(min(count, (flat > 0).sum().item() or 1))
    pairs = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if value <= 0:
            continue
        pairs.append((index // CLASSES, index % CLASSES, value))
    return pairs


# --------------------------------------------------------------------------- #
# Edit primitives -- the moves.                                                #
# --------------------------------------------------------------------------- #
def edit_repurpose(weights, thresholds, layer, neuron, class_a, class_b, class_inputs, config):
    """Rebuild a wasted unit into a discriminator for a confused class pair.

    Its incoming weights become the class-conditional input difference, so it
    fires for class_a and not class_b. This is the edit backprop cannot make: it
    does not reweight an existing feature, it *creates a missing one*.
    """
    new_weights = [w.clone() for w in weights]
    template = class_inputs[layer][class_a] - class_inputs[layer][class_b]
    new_weights[layer][neuron] = normalize_rows(template, config.weight_norm)
    return new_weights, [t.clone() for t in thresholds]


def edit_one_vs_rest(weights, thresholds, layer, neuron, cls, class_inputs, config):
    """Rebuild a wasted unit into a detector for one class against all others."""
    new_weights = [w.clone() for w in weights]
    others = torch.cat([class_inputs[layer][c].unsqueeze(0)
                        for c in range(CLASSES) if c != cls], dim=0).mean(dim=0)
    template = class_inputs[layer][cls] - others
    new_weights[layer][neuron] = normalize_rows(template, config.weight_norm)
    return new_weights, [t.clone() for t in thresholds]


def edit_resample(weights, thresholds, layer, neuron, generator, config):
    """Throw a wasted unit away and draw a fresh random feature. Sometimes the
    right move is not a targeted discriminator but simply new diversity."""
    new_weights = [w.clone() for w in weights]
    fresh = torch.randn(fan_in_of(layer), device=weights[0].device, generator=generator)
    new_weights[layer][neuron] = normalize_rows(fresh, config.weight_norm)
    return new_weights, [t.clone() for t in thresholds]


def edit_split(weights, thresholds, layer, source, target, generator, config):
    """Duplicate an overworked unit into a wasted slot and perturb it, so the two
    copies can specialise. Backprop cannot allocate capacity like this."""
    new_weights = [w.clone() for w in weights]
    noise = torch.randn(fan_in_of(layer), device=weights[0].device, generator=generator)
    clone = new_weights[layer][source] + 0.3 * noise * new_weights[layer][source].norm() / noise.norm()
    new_weights[layer][target] = normalize_rows(clone, config.weight_norm)
    return new_weights, [t.clone() for t in thresholds]


def edit_threshold(weights, thresholds, layer, neuron, scale):
    """Gain surgery on a unit that is silent or saturating."""
    new_thresholds = [t.clone() for t in thresholds]
    new_thresholds[layer][neuron] = (new_thresholds[layer][neuron] * scale).clamp(0.2, 3.0)
    return [w.clone() for w in weights], new_thresholds


def propose(usage, pair_cost, class_inputs, opt):
    """The move set. Every proposal targets capacity the readout is *wasting*
    (least-used units); the moves differ in what they build there. All of them are
    verified before they stick, so a bad move costs an evaluation, never damage."""
    pairs = top_confused_pairs(pair_cost, opt.pairs)
    flat_usage = usage.flatten()
    wasted = flat_usage.topk(opt.neurons, largest=False).indices
    busiest = flat_usage.topk(opt.neurons, largest=True).indices
    candidates = []
    for slot in wasted.tolist():
        layer, neuron = slot // HIDDEN_SIZE, slot % HIDDEN_SIZE
        for class_a, class_b, _ in pairs:                       # fix a confusion
            candidates.append(("pair", layer, neuron, class_a, class_b))
            candidates.append(("pair", layer, neuron, class_b, class_a))
        for cls in range(CLASSES):                              # build a class detector
            candidates.append(("rest", layer, neuron, cls, -1))
        candidates.append(("resample", layer, neuron, -1, -1))  # fresh diversity
        candidates.append(("thresh_up", layer, neuron, -1, -1))
        candidates.append(("thresh_down", layer, neuron, -1, -1))
    for slot in busiest.tolist():                               # relieve an overworked unit
        layer, source = slot // HIDDEN_SIZE, slot % HIDDEN_SIZE
        for target_slot in wasted.tolist():
            t_layer, t_neuron = target_slot // HIDDEN_SIZE, target_slot % HIDDEN_SIZE
            if t_layer == layer and t_neuron != source:
                candidates.append(("split", layer, source, t_neuron, -1))
                break
    return candidates


def realise(weights, thresholds, spec, class_inputs, config, generator):
    kind, layer, first, second, _ = spec
    if kind == "pair":       # first=neuron, second=class_a, spec[4]=class_b
        return edit_repurpose(weights, thresholds, layer, first, second, spec[4],
                              class_inputs, config)
    if kind == "rest":       # first=neuron, second=class
        return edit_one_vs_rest(weights, thresholds, layer, first, second, class_inputs, config)
    if kind == "resample":   # first=neuron
        return edit_resample(weights, thresholds, layer, first, generator, config)
    if kind == "split":      # first=source neuron, second=target neuron
        return edit_split(weights, thresholds, layer, first, second, generator, config)
    if kind == "thresh_up":
        return edit_threshold(weights, thresholds, layer, first, 1.25)
    if kind == "thresh_down":
        return edit_threshold(weights, thresholds, layer, first, 0.8)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# The verified greedy loop                                                     #
# --------------------------------------------------------------------------- #
def architect(weights, thresholds, pool, monitor, config, opt):
    """Verified greedy editing.

    Each round draws a FRESH (fit, verify) split from the training pool. Selecting
    thousands of candidates against one fixed split would simply overfit that
    split; resampling makes every acceptance a fresh out-of-sample test, and an
    edit must clear a margin to be kept. `monitor` is a held-out set used only to
    report honest progress -- it never influences a decision.
    """
    pool_x, pool_y = pool
    accepted, rejected = 0, 0
    start = time.monotonic()
    generator = torch.Generator(device=pool_x.device).manual_seed(opt.seed0 * 31 + 7)

    for round_index in range(opt.rounds):
        # Two independent splits per round. `screen` ranks the candidates;
        # `confirm` -- which played no part in choosing the winner -- decides
        # whether it is kept. Screening alone would just crown whichever candidate
        # got luckiest on the screening split (the winner's curse), which is
        # exactly how the accuracy-based version destroyed the network.
        order = torch.randperm(len(pool_x), device=pool_x.device, generator=generator)
        take = lambda a, b: (pool_x[order[a:b]], pool_y[order[a:b]])
        fit = take(0, opt.fit)
        screen = take(opt.fit, opt.fit + opt.val)
        # Stage 1 only has to RANK, so it may use smaller splits; stage 2 does the
        # statistically serious test. This keeps a large move set affordable.
        screen_fit = take(0, opt.screen_fit)
        screen_val = take(opt.fit, opt.fit + opt.screen_val)
        # The confirm split only has to be independent of the *selection*; the
        # readout may still be fitted on `fit`, since `fit` never ranks anything.
        confirm = take(opt.fit + opt.val, opt.fit + 2 * opt.val)

        base_loss, _, _ = score(weights, thresholds, screen_fit, screen_val, config, opt.ridge)
        _, pair_cost, usage, class_inputs = diagnose(weights, thresholds, fit, screen, config, opt.ridge)
        candidates = propose(usage, pair_cost, class_inputs, opt)
        if not candidates:
            print("  no candidates; stopping")
            break

        # Stage 1 -- screen: rank every candidate by held-out LOSS (lower is better).
        ranked = []
        for spec in candidates:
            trial_w, trial_t = realise(weights, thresholds, spec, class_inputs, config, generator)
            trial_loss, _, _ = score(trial_w, trial_t, screen_fit, screen_val, config, opt.ridge)
            if trial_loss < base_loss:
                ranked.append((trial_loss, spec, trial_w, trial_t))
        ranked.sort(key=lambda item: item[0])

        # Stage 2 -- confirm: the shortlist must prove itself on data that had no
        # say in selecting it, against the same margin.
        taken = False
        confirm_base, _, _ = score(weights, thresholds, fit, confirm, config, opt.ridge)
        for _, spec, trial_w, trial_t in ranked[:opt.shortlist]:
            trial_loss, _, _ = score(trial_w, trial_t, fit, confirm, config, opt.ridge)
            if trial_loss < confirm_base - opt.margin:
                weights, thresholds = trial_w, trial_t
                accepted += 1
                taken = True
                break
        rejected += len(candidates) - (1 if taken else 0)

        if (round_index + 1) % opt.report_every == 0:
            _, held, _ = score(weights, thresholds, (pool_x[:opt.fit], pool_y[:opt.fit]),
                               monitor, config, opt.ridge)
            print(f"  round {round_index + 1:4d}: accepted {accepted:4d} / rejected {rejected:5d}  "
                  f"monitor {100 * held:.2f}%  [{time.monotonic() - start:.0f}s]", flush=True)

    print(f"  edits accepted {accepted}, proposals rejected {rejected}")
    return weights, thresholds, accepted


# --------------------------------------------------------------------------- #
# Final, honest evaluation: readout fit on the FULL train set, scored on TEST. #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def final_report(weights, thresholds, train, test, config, opt, label):
    train_x, train_y = train
    test_x, test_y = test
    train_features, _ = forward(weights, thresholds, train_x, config)
    test_features, _ = forward(weights, thresholds, test_x, config)
    mean, std, readout = fit_ridge(torch.cat(train_features, dim=1), train_y, opt.ridge)
    logits = apply_readout(torch.cat(test_features, dim=1), mean, std, readout)
    concat = accuracy_of(logits, test_y)
    per_layer = []
    for index in range(HIDDEN_LAYERS):
        m, s, r = fit_ridge(train_features[index], train_y, opt.ridge)
        per_layer.append(accuracy_of(apply_readout(test_features[index], m, s, r), test_y))
    layers = "  ".join(f"L{i + 1} {100 * a:.2f}%" for i, a in enumerate(per_layer))
    print(f"  {label:22s} {layers}  concat {100 * concat:.2f}%", flush=True)
    return concat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/kmnist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--neurons", type=int, default=12,
                        help="least-used units considered for repurposing each round")
    parser.add_argument("--pairs", type=int, default=3,
                        help="most-confused class pairs targeted each round")
    parser.add_argument("--fit", type=int, default=4000, help="samples used to fit the readout during search")
    parser.add_argument("--val", type=int, default=3000, help="held-out samples used to VERIFY each edit")
    parser.add_argument("--monitor", type=int, default=5000,
                        help="held out from the search entirely; progress reporting only")
    parser.add_argument("--margin", type=float, default=0.002,
                        help="required LOSS improvement on the confirm split to keep an edit")
    parser.add_argument("--screen-fit", type=int, default=2000,
                        help="smaller readout-fit split used only to RANK candidates (stage 1)")
    parser.add_argument("--screen-val", type=int, default=2000,
                        help="smaller eval split used only to RANK candidates (stage 1)")
    parser.add_argument("--shortlist", type=int, default=3,
                        help="top screened candidates re-tested on the independent confirm split")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--init", choices=["random", "stdp"], default="random",
                        help="edit a random net, or edit ON TOP of an STDP-trained one")
    parser.add_argument("--stdp-epochs", type=int, default=3)
    parser.add_argument("--stdp-batch", type=int, default=128)
    parser.add_argument("--stdp-lr", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=1)
    return parser.parse_args()


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        fail("CUDA is required")
    device = torch.device(opt.device)
    torch.cuda.set_device(device)
    config = Config(timesteps=opt.timesteps, top_k=opt.top_k)

    (train_x, train_y), test = load_split(opt.data, device)
    # Everything the search sees comes from TRAIN. `monitor` is held out from the
    # search entirely (progress reporting only); TEST is untouched until the end.
    monitor = (train_x[:opt.monitor], train_y[:opt.monitor])
    pool = (train_x[opt.monitor:], train_y[opt.monitor:])
    train = (train_x, train_y)
    print(f"kmnist_architect: {len(pool[0])} search pool ({opt.fit} fit / {opt.val} verify, "
          f"resampled each round), {opt.monitor} monitor, {len(test[0])} test, "
          f"device={torch.cuda.get_device_name(device)}")
    print("STDP reference: 73.6% concat   untrained: 66.3%   teacher: 83.6%\n")

    for repeat in range(opt.seeds):
        seed = opt.seed0 + repeat
        generator = torch.Generator(device=device).manual_seed(seed)
        weights, thresholds = init_snn(device, generator, config)
        print(f"seed {seed}:")
        if opt.init == "stdp":
            print("  pre-training with STDP...", flush=True)
            weights, thresholds = train_stdp(weights, thresholds, train_x, config, opt, device)
            before = final_report(weights, thresholds, train, test, config, opt, "STDP (baseline)")
        else:
            before = final_report(weights, thresholds, train, test, config, opt, "untrained")
        weights, thresholds, edits = architect(weights, thresholds, pool, monitor, config, opt)
        after = final_report(weights, thresholds, train, test, config, opt, "after editing")
        print(f"  TEST: {100 * before:.2f}% -> {100 * after:.2f}%  "
              f"({100 * (after - before):+.2f})  from {edits} verified edits\n", flush=True)


if __name__ == "__main__":
    main()
