#!/usr/bin/env python3
"""Pair-STDP experiment for the four-hidden-layer KMNIST SNN.

The hidden network is 784-256-256-256-256. It uses the benchmark's static
pixel drive and trains all four LIF layers with local trace-based STDP. A
closed-form ridge calibration reads ten classes out of the spike statistic of
every hidden layer, of all four concatenated, and (as a control) of the raw
pixels, so the write-up can show what depth does to a linearly-read feature.

This tool deliberately requires CUDA. It is an experiment backend, like
tools/kmnist_cnn.py and tools/bptt_cuda.cu, rather than a public library API.
"""

import argparse
import csv
import gzip
import math
import os
import struct
import sys
import time
from dataclasses import dataclass

# cuBLAS needs this setting for deterministic GEMMs when deterministic
# algorithms are enabled. It must be set before the first CUDA operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 256
HIDDEN_LAYERS = 4
CLASSES = 10


def fail(message):
    raise SystemExit(f"kmnist_stdp: {message}")


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


@dataclass
class STDPConfig:
    timesteps: int = 20
    beta: float = 0.95
    top_k: int = 32
    initial_threshold: float = 1.0
    threshold_lr: float = 0.05
    target_rate: float = 0.08
    threshold_min: float = 0.4
    threshold_max: float = 2.0
    trace_decay: float = 0.5
    ltd_ratio: float = 0.5
    stdp_lr: float = 0.005
    weight_norm: float = 2.0


@dataclass
class LIFLayer:
    weight: torch.Tensor
    threshold: torch.Tensor


class STDPNetwork:
    def __init__(self, config, device, generator):
        self.config = config
        self.device = device
        self.layers = []
        fan_in = INPUT_SIZE
        for _ in range(HIDDEN_LAYERS):
            # QR gives diverse, signed incoming rows without letting a few
            # neurons monopolize the first WTA decisions.
            basis = torch.randn((fan_in, HIDDEN_SIZE), device=device, generator=generator)
            orthogonal, triangular = torch.linalg.qr(basis, mode="reduced")
            sign = triangular.diagonal().sign().masked_fill_(triangular.diagonal().eq(0), 1.0)
            weight = (orthogonal * sign).t().contiguous().mul_(config.weight_norm)
            threshold = torch.full((HIDDEN_SIZE,), config.initial_threshold, device=device)
            self.layers.append(LIFLayer(weight, threshold))
            fan_in = HIDDEN_SIZE

    def assert_cuda(self):
        if self.device.type != "cuda":
            raise AssertionError("network is not on CUDA")
        for layer in self.layers:
            if not layer.weight.is_cuda or not layer.threshold.is_cuda:
                raise AssertionError("STDP parameter escaped CUDA")


def make_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def lif_step(layer, pre_spikes, voltage, previous_spikes, config):
    # This is the benchmark recurrence: u[t] = beta*u[t-1] + drive - theta*s[t-1].
    voltage.mul_(config.beta).addmm_(pre_spikes, layer.weight.t())
    voltage.sub_(previous_spikes * layer.threshold)
    score = voltage - layer.threshold
    winner_score, winner_index = score.topk(config.top_k, dim=1, sorted=False)
    winner_spikes = winner_score.ge(0.0).to(voltage.dtype)
    spikes = torch.zeros_like(voltage)
    spikes.scatter_(1, winner_index, winner_spikes)
    return spikes


def pair_stdp_statistics(pre_tape, post_tape, trace_decay):
    """Return causal and anti-causal pair counts for [T, B, N] spike tapes."""
    if pre_tape.ndim != 3 or post_tape.ndim != 3 or pre_tape.shape[:2] != post_tape.shape[:2]:
        raise ValueError("STDP tapes must be [T, B, N] with matching T and B")
    pre_trace = torch.zeros_like(pre_tape[0])
    post_trace = torch.zeros_like(post_tape[0])
    causal_trace = []
    anti_trace = []
    for timestep in range(pre_tape.shape[0]):
        pre_trace = trace_decay * pre_trace + pre_tape[timestep]
        causal_trace.append(pre_trace)
        anti_trace.append(post_trace)
        post_trace = trace_decay * post_trace + post_tape[timestep]
    flat_post = post_tape.flatten(0, 1)
    flat_pre = pre_tape.flatten(0, 1)
    potentiation = flat_post.t() @ torch.stack(causal_trace).flatten(0, 1)
    depression = torch.stack(anti_trace).flatten(0, 1).t() @ flat_pre
    return potentiation, depression


def apply_stdp(layer, pre_tape, post_tape, config, learning_rate):
    potentiation, depression = pair_stdp_statistics(pre_tape, post_tape, config.trace_decay)
    post_count = post_tape.sum(dim=(0, 1)).unsqueeze(1).clamp_min_(1.0)
    eligibility = (potentiation - config.ltd_ratio * depression) / post_count
    eligibility.sub_(eligibility.mean(dim=1, keepdim=True))
    eligibility.div_(eligibility.norm(dim=1, keepdim=True).clamp_min_(1e-8))
    layer.weight.add_(eligibility, alpha=learning_rate)
    normalize_signed_rows(layer.weight, config.weight_norm)


def normalize_signed_rows(weight, norm):
    weight.sub_(weight.mean(dim=1, keepdim=True))
    weight.mul_(norm / weight.norm(dim=1, keepdim=True).clamp_min_(1e-8))


def simulate_batch(network, images_u8, collect_rates=False):
    """Run frozen hidden layers and return every layer's linear-readout statistic."""
    config = network.config
    batch_size = len(images_u8)
    pixels = images_u8.float().mul_(1.0 / 255.0)
    voltages = [torch.zeros((batch_size, HIDDEN_SIZE), device=network.device) for _ in network.layers]
    previous = [torch.zeros_like(voltages[0]) for _ in network.layers]
    totals = [torch.zeros(HIDDEN_SIZE, device=network.device) for _ in network.layers]
    filtered = [torch.zeros_like(voltages[0]) for _ in network.layers]
    feature_sum = [torch.zeros_like(voltages[0]) for _ in network.layers]

    for _ in range(config.timesteps):
        spikes = pixels
        for layer_index, layer in enumerate(network.layers):
            spikes = lif_step(layer, spikes, voltages[layer_index], previous[layer_index], config)
            previous[layer_index] = spikes
            filtered[layer_index].mul_(config.beta).add_(spikes)
            feature_sum[layer_index].add_(filtered[layer_index])
            if collect_rates:
                totals[layer_index].add_(spikes.sum(dim=0))
    return [total.div_(config.timesteps) for total in feature_sum], totals


def train_stdp_epoch(network, images, order, batch_size, learning_rate, threshold_lr):
    """Train every hidden layer from local traces collected in one forward pass."""
    config = network.config
    spike_totals = [torch.zeros(HIDDEN_SIZE, device=network.device) for _ in network.layers]
    sample_steps = 0

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        pixels = images[indices].float().mul_(1.0 / 255.0)
        count = len(pixels)
        voltages = [torch.zeros((count, HIDDEN_SIZE), device=network.device) for _ in network.layers]
        previous = [torch.zeros_like(voltages[0]) for _ in network.layers]
        pre_traces = [
            torch.zeros((count, INPUT_SIZE if index == 0 else HIDDEN_SIZE), device=network.device)
            for index in range(HIDDEN_LAYERS)
        ]
        post_traces = [torch.zeros_like(voltages[0]) for _ in network.layers]
        potentiation = [torch.zeros_like(layer.weight) for layer in network.layers]
        depression = [torch.zeros_like(layer.weight) for layer in network.layers]
        batch_spikes = [torch.zeros(HIDDEN_SIZE, device=network.device) for _ in network.layers]

        for _ in range(config.timesteps):
            spikes = pixels
            for layer_index, layer in enumerate(network.layers):
                pre = spikes
                spikes = lif_step(layer, pre, voltages[layer_index], previous[layer_index], config)
                pre_traces[layer_index].mul_(config.trace_decay).add_(pre)
                potentiation[layer_index].addmm_(spikes.t(), pre_traces[layer_index])
                depression[layer_index].addmm_(post_traces[layer_index].t(), pre)
                post_traces[layer_index].mul_(config.trace_decay).add_(spikes)
                previous[layer_index] = spikes
                batch_spikes[layer_index].add_(spikes.sum(dim=0))

        for layer_index, layer in enumerate(network.layers):
            eligibility = potentiation[layer_index].add(
                depression[layer_index], alpha=-config.ltd_ratio
            )
            eligibility.div_(batch_spikes[layer_index].unsqueeze(1).clamp_min_(1.0))
            eligibility.sub_(eligibility.mean(dim=1, keepdim=True))
            eligibility.div_(eligibility.norm(dim=1, keepdim=True).clamp_min_(1e-8))
            layer.weight.add_(eligibility, alpha=learning_rate)
            normalize_signed_rows(layer.weight, config.weight_norm)

            rates = batch_spikes[layer_index] / (count * config.timesteps)
            layer.threshold.add_(rates - config.target_rate, alpha=threshold_lr)
            layer.threshold.clamp_(config.threshold_min, config.threshold_max)
            spike_totals[layer_index].add_(batch_spikes[layer_index])
        sample_steps += count * config.timesteps

    return [total / sample_steps for total in spike_totals]


def extract_features(network, images, batch_size):
    """Return per-layer [N, HIDDEN] readout statistics, per-layer rates, and wall time."""
    features = [torch.empty((len(images), HIDDEN_SIZE), device=network.device) for _ in network.layers]
    layer_totals = [torch.zeros(HIDDEN_SIZE, device=network.device) for _ in network.layers]
    start_time = time.monotonic()
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        statistics, totals = simulate_batch(network, batch, collect_rates=True)
        for layer_index in range(HIDDEN_LAYERS):
            features[layer_index][start:start + len(batch)] = statistics[layer_index]
            layer_totals[layer_index].add_(totals[layer_index])
    torch.cuda.synchronize(network.device)
    elapsed = time.monotonic() - start_time
    denominator = len(images) * network.config.timesteps
    rates = [total / denominator for total in layer_totals]
    return features, rates, elapsed


def fit_ridge_readout(features, labels, ridge):
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min_(0.02)
    standardized = (features - mean) / std
    design = torch.cat(
        (standardized, torch.ones((len(features), 1), device=features.device)), dim=1
    )
    targets = torch.nn.functional.one_hot(labels, CLASSES).float()
    gram = design.t() @ design
    gram.diagonal().add_(ridge)
    readout = torch.linalg.solve(gram, design.t() @ targets)
    return mean, std, readout


def ridge_metrics(features, labels, mean, std, readout):
    standardized = (features - mean) / std
    design = torch.cat(
        (standardized, torch.ones((len(features), 1), device=features.device)), dim=1
    )
    logits = design @ readout
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    accuracy = logits.argmax(dim=1).eq(labels).float().mean().item()
    return loss, accuracy


def ridge_readout(train_features, train_labels, test_features, test_labels, ridge):
    """Fit a ridge readout on the train features and score both splits."""
    mean, std, readout = fit_ridge_readout(train_features, train_labels, ridge)
    train_loss, train_accuracy = ridge_metrics(train_features, train_labels, mean, std, readout)
    test_loss, test_accuracy = ridge_metrics(test_features, test_labels, mean, std, readout)
    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def pixel_baseline(train, test, ridge):
    """Linear-readout control: ridge straight off the raw 0-1 pixels, no network."""
    train_x, train_y = train
    test_x, test_y = test
    return ridge_readout(train_x.float(), train_y, test_x.float(), test_y, ridge)


def evaluate_readout(network, train, test, batch_size, ridge):
    """Read every hidden layer, and all four concatenated, out of the frozen net."""
    train_x, train_y = train
    test_x, test_y = test
    train_features, train_rates, train_seconds = extract_features(network, train_x, batch_size)
    test_features, test_rates, test_seconds = extract_features(network, test_x, batch_size)
    layers = [
        ridge_readout(train_features[index], train_y, test_features[index], test_y, ridge)
        for index in range(HIDDEN_LAYERS)
    ]
    concat = ridge_readout(
        torch.cat(train_features, dim=1), train_y, torch.cat(test_features, dim=1), test_y, ridge
    )
    return {
        "layers": layers,
        "concat": concat,
        "train_rates": train_rates,
        "test_rates": test_rates,
        "seconds": train_seconds + test_seconds,
    }


def validate_options(opt):
    if opt.timesteps <= 0 or opt.epochs <= 0 or opt.batch <= 0 or opt.eval_batch <= 0:
        fail("timesteps, epochs, and batch sizes must be positive")
    if not 0.0 < opt.beta < 1.0:
        fail("beta must be in (0,1)")
    if not 1 <= opt.top_k <= HIDDEN_SIZE:
        fail(f"top-k must be in [1,{HIDDEN_SIZE}]")
    if not 0.0 <= opt.trace_decay < 1.0 or opt.stdp_lr <= 0.0 or not 0.0 < opt.lr_decay <= 1.0:
        fail("trace-decay must be in [0,1), stdp-lr positive, and lr-decay in (0,1]")
    if opt.threshold_min <= 0.0 or opt.threshold_max < opt.threshold_min:
        fail("invalid threshold bounds")
    if not opt.threshold_min <= opt.threshold <= opt.threshold_max:
        fail("initial threshold must lie within threshold bounds")
    if opt.seeds <= 0:
        fail("seeds must be positive")
    if opt.update_reference <= 0 or opt.ridge <= 0.0 or opt.weight_norm <= 0.0:
        fail("update-reference, ridge, and weight-norm must be positive")


def config_from_options(opt):
    return STDPConfig(
        timesteps=opt.timesteps,
        beta=opt.beta,
        top_k=opt.top_k,
        initial_threshold=opt.threshold,
        threshold_lr=opt.threshold_lr,
        target_rate=opt.target_rate,
        threshold_min=opt.threshold_min,
        threshold_max=opt.threshold_max,
        trace_decay=opt.trace_decay,
        ltd_ratio=opt.ltd_ratio,
        stdp_lr=opt.stdp_lr,
        weight_norm=opt.weight_norm,
    )


CSV_FIELDS = [
    "tag", "seed", "gpu", "torch", "train_samples", "test_samples", "timesteps", "epochs",
    "stdp_lr", "effective_stdp_lr", "trace_decay", "ridge",
    "pixel_train_acc", "pixel_test_acc",
    "baseline_concat_train_acc", "baseline_concat_test_acc",
    "final_concat_train_acc", "final_concat_test_acc",
] + [f"baseline_layer{i}_test_acc" for i in range(1, HIDDEN_LAYERS + 1)
     ] + [f"final_layer{i}_{metric}" for i in range(1, HIDDEN_LAYERS + 1)
          for metric in ("train_acc", "test_acc")] + [
    "stdp_seconds", "feature_seconds", "peak_vram_mb",
] + [f"layer{i}_{metric}" for i in range(1, HIDDEN_LAYERS + 1)
     for metric in ("train_rate", "test_rate", "dead_fraction", "weight_delta", "threshold")]


def csv_writer(path):
    if not path:
        return None, None
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    stream = open(path, "a", newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    if fresh:
        writer.writeheader()
        stream.flush()
    return stream, writer


def format_layer_line(result):
    return "  ".join(
        f"L{index + 1} {100 * layer['test_accuracy']:.2f}%"
        for index, layer in enumerate(result["layers"])
    ) + f"  concat {100 * result['concat']['test_accuracy']:.2f}%"


@torch.no_grad()
def run_seed(opt, config, device, train, test, seed, pixel):
    initialization_gen = make_generator(device, seed)
    network = STDPNetwork(config, device, initialization_gen)
    network.assert_cuda()
    torch.cuda.reset_peak_memory_stats(device)

    scale = min(1.0, opt.update_reference / len(train[0]))
    effective_lr = config.stdp_lr * scale
    initial_weights = [layer.weight.clone() for layer in network.layers]
    print(f"seed={seed} architecture=784-256-256-256-256-10 T={config.timesteps} "
          f"batch={opt.batch} stdp_lr={config.stdp_lr:g} effective_lr={effective_lr:.7g}")
    baseline = evaluate_readout(network, train, test, opt.eval_batch, opt.ridge)
    print(f"  untrained readout test: {format_layer_line(baseline)}"
          f"   [pixels {100 * pixel['test_accuracy']:.2f}%]", flush=True)

    final_rates = None
    train_start = time.monotonic()
    for epoch in range(opt.epochs):
        order_gen = make_generator(device, seed * 100003 + epoch)
        order = torch.randperm(len(train[0]), device=device, generator=order_gen)
        learning_rate = effective_lr * (opt.lr_decay ** epoch)
        torch.cuda.synchronize(device)
        started = time.monotonic()
        final_rates = train_stdp_epoch(
            network, train[0], order, opt.batch, learning_rate, config.threshold_lr * scale
        )
        torch.cuda.synchronize(device)
        seconds = time.monotonic() - started
        rate_text = "  ".join(f"L{i + 1} {rate.mean().item():.4f}" for i, rate in enumerate(final_rates))
        print(f"  STDP epoch {epoch + 1}: {rate_text}  {seconds:.1f}s", flush=True)
    torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - train_start

    final = evaluate_readout(network, train, test, opt.eval_batch, opt.ridge)
    print(f"  trained readout test:   {format_layer_line(final)}"
          f"   STDP {train_seconds:.1f}s  features {final['seconds']:.1f}s", flush=True)

    row = {
        "tag": opt.tag,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "train_samples": len(train[0]),
        "test_samples": len(test[0]),
        "timesteps": config.timesteps,
        "epochs": opt.epochs,
        "stdp_lr": config.stdp_lr,
        "effective_stdp_lr": effective_lr,
        "trace_decay": config.trace_decay,
        "ridge": opt.ridge,
        "pixel_train_acc": pixel["train_accuracy"],
        "pixel_test_acc": pixel["test_accuracy"],
        "baseline_concat_train_acc": baseline["concat"]["train_accuracy"],
        "baseline_concat_test_acc": baseline["concat"]["test_accuracy"],
        "final_concat_train_acc": final["concat"]["train_accuracy"],
        "final_concat_test_acc": final["concat"]["test_accuracy"],
        "stdp_seconds": train_seconds,
        "feature_seconds": baseline["seconds"] + final["seconds"],
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
    }
    for index in range(HIDDEN_LAYERS):
        row[f"baseline_layer{index + 1}_test_acc"] = baseline["layers"][index]["test_accuracy"]
        row[f"final_layer{index + 1}_train_acc"] = final["layers"][index]["train_accuracy"]
        row[f"final_layer{index + 1}_test_acc"] = final["layers"][index]["test_accuracy"]
    for index, rates in enumerate(final_rates):
        prefix = f"layer{index + 1}_"
        test_rates = final["test_rates"][index]
        row[prefix + "train_rate"] = rates.mean().item()
        row[prefix + "test_rate"] = test_rates.mean().item()
        row[prefix + "dead_fraction"] = test_rates.eq(0).float().mean().item()
        row[prefix + "weight_delta"] = (
            (network.layers[index].weight - initial_weights[index]).norm()
            / initial_weights[index].norm()
        ).item()
        row[prefix + "threshold"] = network.layers[index].threshold.mean().item()
    return row


@torch.no_grad()
def self_test(device):
    print(f"kmnist_stdp self-test: device={device} ({torch.cuda.get_device_name(device)})")
    decay = 0.5

    causal_pre = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    causal_post = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    anti_pre = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    anti_post = torch.tensor([[[1.0]], [[0.0]], [[0.0]]])
    quiet = torch.zeros_like(causal_pre)

    cpu_p, cpu_d = pair_stdp_statistics(causal_pre, causal_post, decay)
    if not cpu_p[0, 0] > 0 or cpu_d.abs().max() != 0:
        raise AssertionError("causal timing did not produce pure potentiation")
    anti_p, anti_d = pair_stdp_statistics(anti_pre, anti_post, decay)
    if anti_p.abs().max() != 0 or not anti_d[0, 0] > 0:
        raise AssertionError("anti-causal timing did not produce pure depression")
    quiet_p, quiet_d = pair_stdp_statistics(quiet, causal_post.new_zeros((3, 1, 1)), decay)
    if quiet_p.abs().max() != 0 or quiet_d.abs().max() != 0:
        raise AssertionError("quiet tapes changed a synapse")

    gpu_p, gpu_d = pair_stdp_statistics(causal_pre.to(device), causal_post.to(device), decay)
    if not torch.allclose(cpu_p, gpu_p.cpu(), atol=1e-7) or not torch.allclose(cpu_d, gpu_d.cpu(), atol=1e-7):
        raise AssertionError("CPU/CUDA STDP statistics disagree")

    config = STDPConfig(stdp_lr=0.1)
    layer = LIFLayer(
        torch.tensor([[0.0, math.sqrt(2.0), -math.sqrt(2.0)]], device=device),
        torch.tensor([config.initial_threshold], device=device),
    )
    apply_stdp(layer, causal_pre.to(device), causal_post.to(device), config, config.stdp_lr)
    if not layer.weight[0, 0] > 0.0:
        raise AssertionError("causal synapse was not potentiated")
    if not torch.allclose(layer.weight.mean(dim=1), torch.zeros(1, device=device), atol=1e-6):
        raise AssertionError("STDP broke incoming-weight centering")
    if not torch.allclose(layer.weight.norm(dim=1), torch.full((1,), 2.0, device=device), atol=1e-6):
        raise AssertionError("STDP broke incoming-weight normalization")

    generator = make_generator(device, 123)
    network = STDPNetwork(config, device, generator)
    network.assert_cuda()
    images = torch.randint(0, 256, (32, INPUT_SIZE), dtype=torch.uint8, device=device, generator=generator)
    initial_weight = [entry.weight.clone() for entry in network.layers]
    rates = train_stdp_epoch(
        network, images, torch.arange(len(images), device=device), len(images), config.stdp_lr, config.threshold_lr
    )
    if any(torch.equal(old, entry.weight) for old, entry in zip(initial_weight, network.layers)):
        raise AssertionError("end-to-end STDP left a hidden layer unchanged")
    if any(not torch.isfinite(entry.weight).all() for entry in network.layers):
        raise AssertionError("end-to-end STDP produced a non-finite weight")
    if any(rate.mean() <= 0 for rate in rates):
        raise AssertionError("end-to-end STDP did not propagate spikes through every layer")

    before_weight = [entry.weight.clone() for entry in network.layers]
    before_threshold = [entry.threshold.clone() for entry in network.layers]
    simulate_batch(network, images, collect_rates=True)
    if any(not torch.equal(old, entry.weight) for old, entry in zip(before_weight, network.layers)):
        raise AssertionError("frozen evaluation changed a weight")
    if any(not torch.equal(old, entry.threshold) for old, entry in zip(before_threshold, network.layers)):
        raise AssertionError("frozen evaluation changed a threshold")
    print("  causal LTP, anti-causal LTD, quiet rule: pass")
    print("  CPU/CUDA correlation parity: pass")
    print("  four-layer CUDA update, normalization, frozen evaluation: pass")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/kmnist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--threshold-lr", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.08)
    parser.add_argument("--threshold-min", type=float, default=0.4)
    parser.add_argument("--threshold-max", type=float, default=2.0)
    parser.add_argument("--trace-decay", type=float, default=0.5)
    parser.add_argument("--ltd-ratio", type=float, default=0.5)
    parser.add_argument("--stdp-lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--weight-norm", type=float, default=2.0)
    parser.add_argument("--update-reference", type=int, default=5000,
                        help="scale per-batch plasticity by min(1, N/number of training samples)")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=1)
    parser.add_argument("--csv")
    parser.add_argument("--tag", default="stdp_d4")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    opt = parse_args()
    validate_options(opt)
    if not torch.cuda.is_available():
        fail("CUDA is required; no CUDA device is available")
    device = torch.device(opt.device)
    if device.type != "cuda":
        fail("training is CUDA-only; --device must name a CUDA device")
    try:
        torch.cuda.set_device(device)
    except (RuntimeError, ValueError) as error:
        fail(str(error))

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if opt.self_test:
        self_test(device)
        return

    config = config_from_options(opt)
    train, test = load_dataset(opt.data, device, opt.train_limit, opt.test_limit)
    print(f"kmnist_stdp: {len(train[0])} train, {len(test[0])} test, "
          f"device={device} ({torch.cuda.get_device_name(device)}), torch={torch.__version__}")
    pixel = pixel_baseline(train, test, opt.ridge)
    print(f"raw-pixel ridge control: train {100 * pixel['train_accuracy']:.2f}%  "
          f"test {100 * pixel['test_accuracy']:.2f}%", flush=True)

    stream, writer = csv_writer(opt.csv)
    try:
        for repeat in range(opt.seeds):
            row = run_seed(opt, config, device, train, test, opt.seed0 + repeat, pixel)
            if writer:
                writer.writerow(row)
                stream.flush()
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
