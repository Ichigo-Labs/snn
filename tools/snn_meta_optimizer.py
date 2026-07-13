#!/usr/bin/env python3
"""GPU-native meta-optimizer for progressively larger spiking networks.

The learned optimizer is an AlphaZero-style residual policy/value network.  It
observes compact, GPU-resident sketches of every SNN layer and emits bounded
coefficients over safe update bases.  Its zero action is *exactly* projected
Adam, so a newly initialized policy starts from a useful optimizer rather than
having to discover gradient descent through an outer loss.

There are two deliberately separate commands:

``verify``
    Run deterministic GPU correctness, optimizer-oracle, safety, and tiny
    end-to-end checks.  It writes a source-hashed manifest only if every check
    passes.

``benchmark``
    Meta-train and compare the learned optimizer with pair-STDP and surrogate
    BPTT while growing SNN width/depth.  It refuses to start unless a current
    verification manifest exists.  There is no bypass flag.

The benchmark is research code, not a claim that software can be proven free
of every possible bug.  The manifest is an auditable pre-experiment gate: any
change to this file invalidates it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


VERIFY_SCHEMA = 1
OBSERVATION_CHANNELS = 32
SKETCH_BINS = 32
ACTION_BASES = 7
ACTION_CHANNELS = ACTION_BASES + 1  # seven residual bases + Adam log-scale
MIN_POLICY_PARAMETERS = 50_000_000
VERIFICATION_CHECKS = (
    "synthetic_reproducibility_and_balance",
    "synthetic_nonlinearity_controls",
    "snn_recurrence_reference",
    "surrogate_finite_difference",
    "pair_stdp_causality",
    "adam_oracle_10_steps",
    "zero_policy_equals_projected_adam_100_steps",
    "trust_region_and_nonfinite_rollback",
    "tiny_meta_backward_and_update",
    "multi_hidden_nonsquare_indexing",
    "tiny_paired_methods_end_to_end",
    "mixed_precision_policy_parity",
    "full_policy_50m_cuda_forward_backward",
)


def require_cuda(device_name: str) -> torch.device:
    """Return a CUDA device or fail before allocating experiment state."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU experiment execution is intentionally disabled")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(f"device must be CUDA, got {device_name!r}")
    torch.cuda.set_device(device)
    return device


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def scalar(value: float, like: Tensor) -> Tensor:
    return torch.full((), value, device=like.device, dtype=like.dtype)


def tensor_rms(value: Tensor, eps: float = 1e-12) -> Tensor:
    # Clamp *before* sqrt: at exact zero this has a finite zero derivative, while
    # adding epsilon would perturb every nonzero norm and make trust clipping
    # subtly non-idempotent.
    return value.square().mean().clamp_min(eps).sqrt()


def finite_tensors(values: Iterable[Tensor]) -> bool:
    # This is used only at an update transaction boundary, where the host sync is
    # intentional.  There are no per-tensor .item() calls in the hot path.
    checks = [torch.isfinite(value).all() for value in values]
    return bool(torch.stack(checks).all()) if checks else True


def clone_detached(values: Sequence[Tensor]) -> tuple[Tensor, ...]:
    return tuple(value.detach().clone() for value in values)


@dataclass(frozen=True)
class TaskConfig:
    input_size: int = 16
    classes: int = 4
    clusters_per_class: int = 8
    centre_radius: float = 3.0
    noise: float = 0.28
    train_samples: int = 2048
    validation_samples: int = 512
    test_samples: int = 512

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.centre_radius, self.noise)):
            raise ValueError("task scale/noise must be finite")
        if self.input_size < 2:
            raise ValueError("input_size must be at least 2")
        if self.classes < 2:
            raise ValueError("classes must be at least 2")
        if self.clusters_per_class < 2 or self.clusters_per_class % 2:
            raise ValueError("clusters_per_class must be a positive even number")
        if self.centre_radius <= 0 or self.noise <= 0:
            raise ValueError("centre_radius and noise must be positive")
        if min(self.train_samples, self.validation_samples, self.test_samples) < self.classes:
            raise ValueError("every split must contain at least one sample per class")


@dataclass(frozen=True)
class SyntheticTask:
    train: tuple[Tensor, Tensor]
    validation: tuple[Tensor, Tensor]
    test: tuple[Tensor, Tensor]
    seed: int


def make_synthetic_task(config: TaskConfig, device: torch.device, seed: int) -> SyntheticTask:
    """Create a balanced nonlinear Gaussian-union task entirely on ``device``.

    Every class owns several disconnected random clusters.  The fixed tanh
    transform uses generator parameters, not statistics from validation/test,
    avoiding the split-specific normalization bug in the older scaling tool.
    """
    config.validate()
    gen = cuda_generator(device, seed)
    cluster_count = config.classes * config.clusters_per_class
    # Every class receives direction pairs (+c, -c), so its exact population
    # mean is the same.  This removes the accidental first-order class signal
    # that a finite set of independently random centres can expose to a linear
    # classifier, while an MLP can still identify the disconnected clusters.
    half = config.clusters_per_class // 2
    base = torch.randn(half, config.classes, config.input_size,
                       device=device, generator=gen)
    base = F.normalize(base, dim=2) * config.centre_radius
    centres = torch.cat((base, -base), dim=0).reshape(cluster_count, config.input_size)
    cluster_labels = torch.arange(config.classes, device=device).repeat(config.clusters_per_class)

    def split(count: int, split_seed: int) -> tuple[Tensor, Tensor]:
        split_gen = cuda_generator(device, split_seed)
        # Cycle through clusters before shuffling.  This makes label balance a
        # construction invariant rather than a high-probability property.
        cluster_ids = torch.arange(count, device=device) % cluster_count
        cluster_ids = cluster_ids[torch.randperm(count, device=device, generator=split_gen)]
        noise = torch.randn(count, config.input_size, device=device, generator=split_gen)
        raw = centres[cluster_ids] + config.noise * noise
        inputs = 0.5 * (torch.tanh(raw / config.centre_radius) + 1.0)
        return inputs.contiguous(), cluster_labels[cluster_ids].contiguous()

    return SyntheticTask(
        train=split(config.train_samples, seed * 17 + 1),
        validation=split(config.validation_samples, seed * 17 + 2),
        test=split(config.test_samples, seed * 17 + 3),
        seed=seed,
    )


@dataclass(frozen=True)
class SNNConfig:
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int
    timesteps: int = 8
    beta: float = 0.90
    threshold: float = 0.55
    surrogate_alpha: float = 2.0
    input_scale: float = 2.0
    trace_decay: float = 0.8
    init_gain: float = 1.4

    def validate(self) -> None:
        floats = (self.beta, self.threshold, self.surrogate_alpha,
                  self.input_scale, self.trace_decay, self.init_gain)
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("all SNN hyperparameters must be finite")
        sizes = (self.input_size,) + self.hidden_sizes + (self.output_size,)
        if len(self.hidden_sizes) < 1 or any(size < 1 for size in sizes):
            raise ValueError("an SNN needs positive input, hidden, and output sizes")
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if (self.threshold <= 0 or self.surrogate_alpha <= 0
                or self.input_scale <= 0 or self.init_gain <= 0):
            raise ValueError("threshold, surrogate/input scales, and init_gain must be positive")
        if not 0 <= self.trace_decay < 1:
            raise ValueError("trace_decay must be in [0, 1)")

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.input_size,) + self.hidden_sizes + (self.output_size,)

    @property
    def trainable_layers(self) -> int:
        return len(self.layer_sizes) - 1

    @property
    def parameter_count(self) -> int:
        return sum(out_size * in_size + out_size
                   for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]))


class SurrogateSpike(torch.autograd.Function):
    """Hard Heaviside forward with a peak-normalized ATan surrogate backward."""

    @staticmethod
    def forward(ctx, centered: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(centered)
        ctx.alpha = alpha
        return (centered >= 0).to(centered.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        (centered,) = ctx.saved_tensors
        derivative = 1.0 / (1.0 + (ctx.alpha * centered).square())
        return grad_output * derivative, None


def soft_spike(centered: Tensor, alpha: float) -> Tensor:
    """Primitive whose exact derivative equals ``SurrogateSpike.backward``."""
    return 0.5 + torch.atan(alpha * centered) / alpha


def initialize_snn(config: SNNConfig, device: torch.device, seed: int) -> tuple[Tensor, ...]:
    config.validate()
    gen = cuda_generator(device, seed)
    parameters: list[Tensor] = []
    for fan_in, fan_out in zip(config.layer_sizes[:-1], config.layer_sizes[1:]):
        bound = config.init_gain * math.sqrt(3.0 / fan_in)
        weight = torch.empty(fan_out, fan_in, device=device)
        weight.uniform_(-bound, bound, generator=gen)
        bias = torch.zeros(fan_out, device=device)
        parameters.extend((weight, bias))
    return tuple(parameters)


def weights_of(parameters: Sequence[Tensor]) -> tuple[Tensor, ...]:
    return tuple(parameters[0::2])


def biases_of(parameters: Sequence[Tensor]) -> tuple[Tensor, ...]:
    return tuple(parameters[1::2])


@dataclass(frozen=True)
class SNNTrace:
    pre_rates: tuple[Tensor, ...]
    post_rates: tuple[Tensor, ...]
    margins: tuple[Tensor, ...]
    eligibility: tuple[Tensor, ...]
    mean_spike_rate: Tensor
    max_abs_voltage: Tensor

    def detached(self) -> "SNNTrace":
        return SNNTrace(
            pre_rates=tuple(value.detach() for value in self.pre_rates),
            post_rates=tuple(value.detach() for value in self.post_rates),
            margins=tuple(value.detach() for value in self.margins),
            eligibility=tuple(value.detach() for value in self.eligibility),
            mean_spike_rate=self.mean_spike_rate.detach(),
            max_abs_voltage=self.max_abs_voltage.detach(),
        )


@dataclass(frozen=True)
class SNNResult:
    logits: Tensor
    trace: SNNTrace


def snn_forward(
    parameters: Sequence[Tensor],
    inputs: Tensor,
    config: SNNConfig,
    *,
    spike_mode: str = "surrogate",
    collect_eligibility: bool = True,
) -> SNNResult:
    """Run the one canonical PyTorch SNN recurrence used by every method."""
    if spike_mode not in {"hard", "surrogate", "soft"}:
        raise ValueError(f"unknown spike mode {spike_mode!r}")
    if len(parameters) != 2 * config.trainable_layers:
        raise ValueError("parameter list does not match SNN architecture")
    if inputs.ndim != 2 or inputs.shape[1] != config.input_size:
        raise ValueError("inputs must have shape [batch, input_size]")
    if inputs.device.type != "cuda":
        raise ValueError("SNN execution must stay on CUDA")

    weights = weights_of(parameters)
    biases = biases_of(parameters)
    hidden_count = len(config.hidden_sizes)
    batch = inputs.shape[0]
    dtype, device = inputs.dtype, inputs.device
    membranes = [torch.zeros(batch, size, device=device, dtype=dtype)
                 for size in config.hidden_sizes]
    previous_spikes = [torch.zeros_like(value) for value in membranes]
    spike_sums = [torch.zeros_like(value) for value in membranes]
    margin_sums = [torch.zeros(size, device=device, dtype=dtype) for size in config.hidden_sizes]
    pre_sums = [torch.zeros(size, device=device, dtype=dtype)
                for size in config.layer_sizes[:-1]]
    output_voltage = torch.zeros(batch, config.output_size, device=device, dtype=dtype)
    output_sum = torch.zeros_like(output_voltage)
    output_margin_sum = torch.zeros(config.output_size, device=device, dtype=dtype)
    if collect_eligibility:
        eligibility = [torch.zeros_like(weight) for weight in weights]
        pre_traces = [torch.zeros(batch, size, device=device, dtype=dtype)
                      for size in config.layer_sizes[:-1]]
        post_traces = [torch.zeros(batch, size, device=device, dtype=dtype)
                       for size in config.layer_sizes[1:-1]]
    else:
        eligibility = []
        pre_traces = []
        post_traces = []
    max_abs_voltage = torch.zeros((), device=device, dtype=dtype)

    for _ in range(config.timesteps):
        signal = inputs * config.input_scale
        for layer in range(hidden_count):
            pre_sums[layer] = pre_sums[layer] + signal.detach().abs().sum(dim=0)
            membrane = (config.beta * membranes[layer]
                        + F.linear(signal, weights[layer], biases[layer])
                        - previous_spikes[layer] * config.threshold)
            centered = membrane - config.threshold
            if spike_mode == "surrogate":
                spikes = SurrogateSpike.apply(centered, config.surrogate_alpha)
            elif spike_mode == "soft":
                spikes = soft_spike(centered, config.surrogate_alpha)
            else:
                spikes = (centered >= 0).to(dtype)
            if collect_eligibility:
                # Eligibility is an observation/local-rule statistic, not a path
                # for the meta-gradient.  Detaching it avoids retaining a second
                # copy of the unrolled SNN graph merely to feed the policy.
                pre_event = signal.detach()
                post_event = spikes.detach()
                pre_traces[layer] = config.trace_decay * pre_traces[layer] + pre_event
                eligibility[layer] = (eligibility[layer]
                                      + post_event.transpose(0, 1) @ pre_traces[layer]
                                      - post_traces[layer].transpose(0, 1) @ pre_event)
                post_traces[layer] = config.trace_decay * post_traces[layer] + post_event
            membranes[layer] = membrane
            previous_spikes[layer] = spikes
            spike_sums[layer] = spike_sums[layer] + spikes.detach()
            margin_sums[layer] = margin_sums[layer] + centered.detach().mean(dim=0)
            max_abs_voltage = torch.maximum(max_abs_voltage, membrane.detach().abs().amax())
            signal = spikes

        # The output is a non-spiking leaky integrator, the same semantics as the
        # repository's validated surrogate-BPTT implementation.
        pre_sums[-1] = pre_sums[-1] + signal.detach().abs().sum(dim=0)
        output_voltage = config.beta * output_voltage + F.linear(signal, weights[-1], biases[-1])
        output_sum = output_sum + output_voltage
        output_margin_sum = output_margin_sum + output_voltage.detach().mean(dim=0)
        max_abs_voltage = torch.maximum(max_abs_voltage, output_voltage.detach().abs().amax())

    denom = float(batch * config.timesteps)
    hidden_rates = tuple(total / config.timesteps for total in spike_sums)
    pre_rates = tuple(total / denom for total in pre_sums)
    logits = output_sum / config.timesteps
    output_rate = logits.softmax(dim=1).mean(dim=0)
    post_rates = tuple(rate.mean(dim=0) for rate in hidden_rates) + (output_rate,)
    margins = tuple(total / config.timesteps for total in margin_sums) + (
        output_margin_sum / config.timesteps,
    )
    if collect_eligibility:
        eligibility[-1] = torch.zeros_like(weights[-1])
        eligibility_values = tuple(value / denom for value in eligibility)
    else:
        # Preserve one placeholder per layer without allocating weight-shaped
        # matrices that BPTT/evaluation never consume.
        eligibility_values = tuple(torch.empty(0, device=device, dtype=dtype)
                                   for _ in weights)
    mean_rate = (torch.cat([rate.reshape(-1) for rate in hidden_rates]).mean()
                 if hidden_rates else torch.zeros((), device=device, dtype=dtype))
    return SNNResult(
        logits=logits,
        trace=SNNTrace(pre_rates, post_rates, margins, eligibility_values,
                       mean_rate, max_abs_voltage),
    )


def loss_and_gradients(
    parameters: Sequence[Tensor],
    batch: tuple[Tensor, Tensor],
    config: SNNConfig,
    *,
    create_graph: bool = False,
    collect_eligibility: bool = True,
) -> tuple[Tensor, tuple[Tensor, ...], SNNTrace]:
    leaves = tuple(value if value.requires_grad else value.detach().requires_grad_(True)
                   for value in parameters)
    result = snn_forward(leaves, batch[0], config, spike_mode="surrogate",
                         collect_eligibility=collect_eligibility)
    loss = F.cross_entropy(result.logits, batch[1])
    gradients = torch.autograd.grad(loss, leaves, create_graph=create_graph)
    return loss, tuple(gradient.detach() if not create_graph else gradient
                       for gradient in gradients), result.trace.detached()


def evaluate_snn(
    parameters: Sequence[Tensor],
    split: tuple[Tensor, Tensor],
    config: SNNConfig,
    batch_size: int = 1024,
) -> dict[str, float]:
    losses: list[Tensor] = []
    correct = torch.zeros((), device=split[0].device)
    with torch.no_grad():
        for start in range(0, len(split[0]), batch_size):
            inputs = split[0][start:start + batch_size]
            labels = split[1][start:start + batch_size]
            result = snn_forward(parameters, inputs, config, spike_mode="hard",
                                 collect_eligibility=False)
            losses.append(F.cross_entropy(result.logits, labels, reduction="sum"))
            correct = correct + result.logits.argmax(dim=1).eq(labels).sum()
    count = len(split[0])
    return {"loss": (torch.stack(losses).sum() / count).item(),
            "accuracy": (correct / count).item()}


@torch.no_grad()
def loss_only(parameters: Sequence[Tensor], batch: tuple[Tensor, Tensor],
              config: SNNConfig) -> Tensor:
    result = snn_forward(parameters, batch[0], config, spike_mode="hard",
                         collect_eligibility=False)
    return F.cross_entropy(result.logits, batch[1])


@dataclass(frozen=True)
class PolicyConfig:
    channels: int = 256
    blocks: int = 43
    groups: int = 32
    sketch_bins: int = SKETCH_BINS
    checkpoint_blocks: bool = True

    def validate(self) -> None:
        if self.channels < 8 or self.blocks < 1 or self.sketch_bins < 4 or self.groups < 1:
            raise ValueError("policy channels/blocks/sketch_bins are too small")
        if self.channels % self.groups:
            raise ValueError("policy channels must be divisible by GroupNorm groups")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = F.silu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return F.silu(value + residual)


@dataclass(frozen=True)
class PolicyOutput:
    controls: Tensor  # [batch, depth, ACTION_CHANNELS]
    values: Tensor    # [batch]


class AlphaZeroPolicyOptimizer(nn.Module):
    """Fully convolutional 50M+ policy/value tower over an optimizer board."""

    def __init__(self, config: PolicyConfig = PolicyConfig(), *, enforce_floor: bool = True):
        super().__init__()
        config.validate()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(OBSERVATION_CHANNELS, config.channels, 3, padding=1, bias=False),
            nn.GroupNorm(config.groups, config.channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.channels, config.groups) for _ in range(config.blocks)
        )
        actor_width = 32
        self.actor_head = nn.Sequential(
            nn.Conv2d(config.channels, actor_width, 1, bias=False),
            nn.GroupNorm(8, actor_width),
            nn.SiLU(),
            nn.Conv2d(actor_width, ACTION_CHANNELS, 1, bias=True),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(config.channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.value_mlp = nn.Sequential(nn.Linear(32, 256), nn.SiLU(), nn.Linear(256, 1))
        # The untrained actor is exactly Adam.  Do not zero the shared trunk or
        # value head: their representations/value gradients should be healthy.
        nn.init.zeros_(self.actor_head[-1].weight)
        nn.init.zeros_(self.actor_head[-1].bias)

        if enforce_floor and self.action_parameter_count() < MIN_POLICY_PARAMETERS:
            raise AssertionError(
                f"action path has {self.action_parameter_count():,} params; need >=50M"
            )

    def forward(self, board: Tensor, layer_mask: Tensor | None = None) -> PolicyOutput:
        value = self.stem(board)
        for block in self.blocks:
            if self.config.checkpoint_blocks and self.training and value.requires_grad:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        # Pool only over sketch bins.  The network remains fully convolutional in
        # SNN depth and therefore has no baked-in depth bound.
        controls = self.actor_head(value).mean(dim=3).transpose(1, 2)
        encoded_value = self.value_conv(value).mean(dim=3).transpose(1, 2)
        if layer_mask is None:
            pooled = encoded_value.mean(dim=1)
        else:
            mask = layer_mask.unsqueeze(-1).to(encoded_value.dtype)
            pooled = (encoded_value * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            controls = controls * mask
        values = torch.tanh(self.value_mlp(pooled).squeeze(-1))
        return PolicyOutput(controls.float(), values.float())

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def action_parameter_count(self) -> int:
        value_ids = {id(parameter) for parameter in self.value_conv.parameters()}
        value_ids.update(id(parameter) for parameter in self.value_mlp.parameters())
        return sum(parameter.numel() for parameter in self.parameters()
                   if id(parameter) not in value_ids)


@dataclass(frozen=True)
class Feedback:
    loss: Tensor
    loss_ema: Tensor
    reward: Tensor
    loss_slope: Tensor
    accepted: Tensor
    rejected: Tensor
    rollback: Tensor
    progress: Tensor

    @classmethod
    def initial(cls, loss: Tensor) -> "Feedback":
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
        one = torch.ones((), device=loss.device, dtype=loss.dtype)
        return cls(loss.detach(), loss.detach(), zero, zero, one, zero, zero, zero)


@dataclass(frozen=True)
class OptimizerState:
    momentum: tuple[Tensor, ...]
    variance: tuple[Tensor, ...]
    previous_updates: tuple[Tensor, ...]
    step: int
    feedback: Feedback

    @classmethod
    def zeros(cls, parameters: Sequence[Tensor], initial_loss: Tensor) -> "OptimizerState":
        zeros = tuple(torch.zeros_like(value) for value in parameters)
        return cls(zeros, tuple(value.clone() for value in zeros),
                   tuple(value.clone() for value in zeros), 0, Feedback.initial(initial_loss))

    def detached(self) -> "OptimizerState":
        return OptimizerState(
            tuple(value.detach() for value in self.momentum),
            tuple(value.detach() for value in self.variance),
            tuple(value.detach() for value in self.previous_updates),
            self.step,
            dataclasses.replace(self.feedback,
                                loss=self.feedback.loss.detach(),
                                loss_ema=self.feedback.loss_ema.detach(),
                                reward=self.feedback.reward.detach(),
                                loss_slope=self.feedback.loss_slope.detach(),
                                accepted=self.feedback.accepted.detach(),
                                rejected=self.feedback.rejected.detach(),
                                rollback=self.feedback.rollback.detach(),
                                progress=self.feedback.progress.detach()),
        )


@dataclass(frozen=True)
class AdamProposal:
    updates: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]
    variance: tuple[Tensor, ...]


def adam_proposal(
    gradients: Sequence[Tensor],
    state: OptimizerState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamProposal:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    step = state.step + 1
    momentum = tuple(beta1 * old + (1.0 - beta1) * grad
                     for old, grad in zip(state.momentum, gradients))
    variance = tuple(beta2 * old + (1.0 - beta2) * grad.square()
                     for old, grad in zip(state.variance, gradients))
    correction1 = 1.0 - beta1 ** step
    correction2 = 1.0 - beta2 ** step
    updates = tuple(-learning_rate * (m / correction1)
                    / ((v / correction2).sqrt() + epsilon)
                    for m, v in zip(momentum, variance))
    return AdamProposal(updates, momentum, variance)


def sampled_quantiles(value: Tensor, bins: int, max_samples: int = 4096) -> Tensor:
    """Deterministic GPU quantile sketch with bounded sort cost."""
    flat = value.detach().reshape(-1).float()
    if flat.numel() == 0:
        return torch.zeros(bins, device=value.device)
    if flat.numel() > max_samples:
        sample_indices = torch.linspace(0, flat.numel() - 1, max_samples,
                                        device=value.device).round().long()
        flat = flat.index_select(0, sample_indices)
    ordered = flat.sort().values
    indices = torch.linspace(0, ordered.numel() - 1, bins,
                             device=value.device).round().long()
    return ordered.index_select(0, indices)


def normalized_sketch(value: Tensor, bins: int) -> Tensor:
    sketch = sampled_quantiles(value, bins)
    return sketch / tensor_rms(value.detach().float()).clamp_min(1e-8)


def broadcast_sketch(value: Tensor, bins: int) -> Tensor:
    return value.detach().float().reshape(()).expand(bins)


def build_observation_board(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    adam: AdamProposal,
    state: OptimizerState,
    trace: SNNTrace,
    bins: int = SKETCH_BINS,
) -> tuple[Tensor, Tensor]:
    """Build [1, 32, depth, bins] without moving any hot-path data to CPU."""
    weights = weights_of(parameters)
    weight_gradients = weights_of(gradients)
    adam_weights = weights_of(adam.updates)
    momentum_weights = weights_of(adam.momentum)
    variance_weights = weights_of(adam.variance)
    previous_weights = weights_of(state.previous_updates)
    layers: list[Tensor] = []
    depth = len(weights)
    for index, (weight, grad, adam_update, moment, variance, previous) in enumerate(zip(
            weights, weight_gradients, adam_weights, momentum_weights,
            variance_weights, previous_weights)):
        eligibility = trace.eligibility[index]
        row_norms = weight.detach().float().norm(dim=1)
        grad_row_norms = grad.detach().float().norm(dim=1)
        feedback = state.feedback
        channels = [
            normalized_sketch(weight, bins),
            normalized_sketch(weight.abs(), bins),
            normalized_sketch(grad, bins),
            normalized_sketch(grad.abs(), bins),
            normalized_sketch(moment, bins),
            normalized_sketch(variance.sqrt(), bins),
            normalized_sketch(adam_update, bins),
            normalized_sketch(adam_update.abs(), bins),
            normalized_sketch(eligibility, bins),
            normalized_sketch(eligibility.abs(), bins),
            normalized_sketch(previous, bins),
            normalized_sketch(row_norms, bins),
            normalized_sketch(grad_row_norms, bins),
            normalized_sketch(trace.pre_rates[index], bins),
            normalized_sketch(trace.post_rates[index], bins),
            normalized_sketch(trace.margins[index], bins),
            broadcast_sketch(feedback.loss.log().clamp(-10, 10), bins),
            broadcast_sketch(feedback.loss_ema.log().clamp(-10, 10), bins),
            broadcast_sketch(feedback.reward.clamp(-1, 1), bins),
            broadcast_sketch(feedback.loss_slope.clamp(-1, 1), bins),
            broadcast_sketch(feedback.accepted, bins),
            broadcast_sketch(feedback.rejected, bins),
            broadcast_sketch(feedback.rollback, bins),
            broadcast_sketch(feedback.progress.clamp(0, 1), bins),
            broadcast_sketch(scalar(index / max(1, depth - 1), weight), bins),
            broadcast_sketch(scalar(math.log1p(weight.shape[1]) / 10.0, weight), bins),
            broadcast_sketch(scalar(math.log1p(weight.shape[0]) / 10.0, weight), bins),
            broadcast_sketch(trace.mean_spike_rate, bins),
            broadcast_sketch(tensor_rms(grad).log().clamp(-20, 10), bins),
            broadcast_sketch(tensor_rms(adam_update).log().clamp(-20, 10), bins),
            broadcast_sketch(tensor_rms(weight).log().clamp(-20, 10), bins),
            torch.ones(bins, device=weight.device),
        ]
        if len(channels) != OBSERVATION_CHANNELS:
            raise AssertionError("observation channel accounting error")
        layers.append(torch.stack(channels))
    board = torch.stack(layers, dim=1).unsqueeze(0).contiguous()
    mask = torch.ones(1, depth, device=board.device, dtype=board.dtype)
    return board, mask


@dataclass(frozen=True)
class SafetyConfig:
    max_policy_residual_ratio: float = 0.10
    max_parameter_update_ratio: float = 0.05
    max_element_update: float = 0.5
    max_weight_abs: float = 8.0
    max_bias_abs: float = 4.0
    max_weight_row_norm: float = 8.0
    max_loss_increase_ratio: float = 0.20
    max_loss_increase_absolute: float = 0.20
    min_spike_rate: float = 0.0001
    max_spike_rate: float = 0.80
    max_abs_voltage: float = 100.0

    def validate(self) -> None:
        values = tuple(getattr(self, field.name) for field in dataclasses.fields(self))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all safety limits must be finite")
        if not 0 <= self.max_policy_residual_ratio <= 1:
            raise ValueError("max_policy_residual_ratio must be in [0, 1]")
        if (self.max_parameter_update_ratio <= 0 or self.max_element_update <= 0
                or self.max_weight_abs <= 0 or self.max_bias_abs <= 0
                or self.max_weight_row_norm <= 0 or self.max_abs_voltage <= 0):
            raise ValueError("update limits must be positive")
        if self.max_loss_increase_ratio < 0 or self.max_loss_increase_absolute < 0:
            raise ValueError("loss-increase tolerances cannot be negative")
        if not 0 <= self.min_spike_rate < self.max_spike_rate <= 1:
            raise ValueError("invalid spike-rate safety interval")


@dataclass(frozen=True)
class LearnedProposal:
    parameters: tuple[Tensor, ...]
    raw_updates: tuple[Tensor, ...]
    base_parameters: tuple[Tensor, ...]
    base_updates: tuple[Tensor, ...]
    previous_state: OptimizerState
    next_state: OptimizerState
    predicted_value: Tensor
    trust_penalty: Tensor


def _matched_basis(value: Tensor, reference: Tensor) -> Tensor:
    return value / tensor_rms(value).clamp_min(1e-8) * tensor_rms(reference)


def _row_normalized_gradient(gradient: Tensor) -> Tensor:
    return gradient / gradient.square().mean(dim=1, keepdim=True).add(1e-12).sqrt()


def _column_normalized_gradient(gradient: Tensor) -> Tensor:
    return gradient / gradient.square().mean(dim=0, keepdim=True).add(1e-12).sqrt()


def _clip_update(update: Tensor, parameter: Tensor, safety: SafetyConfig) -> Tensor:
    max_rms = safety.max_parameter_update_ratio * tensor_rms(parameter).clamp_min(1e-3)
    update_rms = tensor_rms(update)
    # The tolerance makes this projection idempotent at its floating-point
    # boundary, which is load-bearing for exact zero-policy == projected-Adam.
    scale = torch.where(update_rms <= max_rms * (1.0 + 1e-6),
                        torch.ones_like(max_rms), max_rms / update_rms)
    return (update * scale).clamp(-safety.max_element_update, safety.max_element_update)


def _project_parameters(parameters: Sequence[Tensor], safety: SafetyConfig) -> tuple[Tensor, ...]:
    projected = []
    for index, value in enumerate(parameters):
        if index % 2 == 0:
            bounded = value.clamp(-safety.max_weight_abs, safety.max_weight_abs)
            row_norm = bounded.norm(dim=1, keepdim=True)
            row_scale = torch.minimum(
                torch.ones_like(row_norm),
                scalar(safety.max_weight_row_norm, bounded) / row_norm.clamp_min(1e-12),
            )
            projected.append(bounded * row_scale)
        else:
            projected.append(value.clamp(-safety.max_bias_abs, safety.max_bias_abs))
    return tuple(projected)


def propose_learned_update(
    policy: AlphaZeroPolicyOptimizer,
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    trace: SNNTrace,
    state: OptimizerState,
    learning_rate: float,
    safety: SafetyConfig,
    *,
    amp: bool = True,
    residual_scale: float = 1.0,
) -> LearnedProposal:
    """Return a differentiable policy candidate and its exact Adam fallback."""
    safety.validate()
    if not finite_tensors((*parameters, *gradients,
                           *state.momentum, *state.variance)):
        raise FloatingPointError("non-finite optimizee gradient/parameter/Adam state")
    adam = adam_proposal(gradients, state, learning_rate)
    board, mask = build_observation_board(parameters, gradients, adam, state, trace,
                                          bins=policy.config.sketch_bins)
    amp_context: contextlib.AbstractContextManager
    amp_context = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp)
    with amp_context:
        output = policy(board, mask)
    controls = output.controls[0]
    new_values: list[Tensor] = []
    raw_updates: list[Tensor] = []
    base_values: list[Tensor] = []
    trust_terms: list[Tensor] = []
    weight_layer = 0
    eligibility_by_parameter = tuple(
        value
        for layer_eligibility, bias in zip(trace.eligibility, biases_of(parameters))
        for value in (layer_eligibility, torch.zeros_like(bias))
    )
    for index, (parameter, gradient, base, moment, eligibility) in enumerate(zip(
            parameters, gradients, adam.updates, adam.momentum,
            eligibility_by_parameter)):
        # Biases deliberately stay on exact Adam.  The expensive shared policy
        # controls every weight matrix, where almost all optimizee parameters and
        # representational capacity live.
        if index % 2:
            update = _clip_update(base, parameter, safety)
        else:
            base_safe = _clip_update(base, parameter, safety)
            action = controls[weight_layer]
            weight_layer += 1
            bases = (
                _matched_basis(-gradient, base_safe),
                _matched_basis(-gradient.sign(), base_safe),
                _matched_basis(-moment, base_safe),
                _matched_basis(eligibility, base_safe),
                _matched_basis(-parameter, base_safe),
                _matched_basis(-_row_normalized_gradient(gradient), base_safe),
                _matched_basis(-_column_normalized_gradient(gradient), base_safe),
            )
            coefficients = torch.tanh(action[:ACTION_BASES])
            residual = sum(coefficient * basis
                           for coefficient, basis in zip(coefficients, bases))
            residual = residual_scale * residual
            base_scaled = base_safe * torch.exp(0.25 * torch.tanh(action[-1]))
            deviation = base_scaled + residual - base_safe
            max_deviation = (safety.max_policy_residual_ratio
                             * tensor_rms(base_safe))
            deviation = deviation * torch.minimum(
                torch.ones_like(max_deviation),
                max_deviation / tensor_rms(deviation).clamp_min(1e-12),
            )
            update = _clip_update(base_safe + deviation, parameter, safety)
            trust_terms.append((update - base_safe).square().mean()
                               / base_safe.square().mean().clamp_min(1e-12))
        raw_updates.append(update)
        new_values.append(parameter + update)
        base_update = _clip_update(base, parameter, safety)
        base_values.append(parameter + base_update)

    projected = _project_parameters(new_values, safety)
    base_projected = _project_parameters(base_values, safety)
    committed_updates = tuple(new - old for new, old in zip(projected, parameters))
    base_committed_updates = tuple(new - old for new, old in zip(base_projected, parameters))
    zero = torch.zeros((), device=parameters[0].device)
    trust_penalty = torch.stack(trust_terms).mean() if trust_terms else zero
    next_feedback = state.feedback
    next_state = OptimizerState(
        tuple(value.detach() for value in adam.momentum),
        tuple(value.detach() for value in adam.variance),
        tuple(value.detach() for value in committed_updates),
        state.step + 1,
        next_feedback,
    )
    return LearnedProposal(projected, committed_updates, base_projected,
                           base_committed_updates,
                           state, next_state, output.values.mean(), trust_penalty)


@dataclass(frozen=True)
class GuardResult:
    parameters: tuple[Tensor, ...]
    state: OptimizerState
    accepted_policy: bool
    used_adam_fallback: bool
    rolled_back: bool
    reason: str
    before_loss: Tensor
    after_loss: Tensor
    reward: Tensor
    trace: SNNTrace


def _guard_metrics(
    parameters: Sequence[Tensor],
    batch: tuple[Tensor, Tensor],
    config: SNNConfig,
) -> tuple[Tensor, SNNTrace]:
    result = snn_forward(parameters, batch[0], config, spike_mode="hard",
                         collect_eligibility=False)
    return F.cross_entropy(result.logits, batch[1]), result.trace


def _candidate_is_safe(
    parameters: Sequence[Tensor],
    loss: Tensor,
    trace: SNNTrace,
    before_loss: Tensor,
    safety: SafetyConfig,
) -> tuple[bool, str]:
    if not finite_tensors((*parameters, loss, trace.mean_spike_rate, trace.max_abs_voltage)):
        return False, "nonfinite"
    bound_checks: list[Tensor] = []
    for index, parameter in enumerate(parameters):
        limit = safety.max_weight_abs if index % 2 == 0 else safety.max_bias_abs
        bound_checks.append(parameter.detach().abs().amax() <= limit + 1e-6)
        if index % 2 == 0:
            bound_checks.append(parameter.detach().norm(dim=1).amax()
                                <= safety.max_weight_row_norm + 1e-6)
    if not bool(torch.stack(bound_checks).all()):
        return False, "parameter_bounds"
    allowed = before_loss * (1.0 + safety.max_loss_increase_ratio) + safety.max_loss_increase_absolute
    if bool(loss.detach() > allowed.detach()):
        return False, "loss_regression"
    hidden_layer_rates = torch.stack(
        [rates.detach().mean() for rates in trace.post_rates[:-1]])
    if bool(((hidden_layer_rates < safety.min_spike_rate)
             | (hidden_layer_rates > safety.max_spike_rate)).any()):
        return False, "firing_rate"
    if bool(trace.max_abs_voltage.detach() > safety.max_abs_voltage):
        return False, "voltage"
    return True, "accepted"


def guarded_commit(
    current_parameters: Sequence[Tensor],
    proposal: LearnedProposal,
    guard_batch: tuple[Tensor, Tensor],
    config: SNNConfig,
    safety: SafetyConfig,
    *,
    before_loss: Tensor | None = None,
    progress: float = 0.0,
) -> GuardResult:
    """Atomically commit policy candidate, safe Adam fallback, or no-op."""
    with torch.no_grad():
        if before_loss is None:
            before_loss, _ = _guard_metrics(current_parameters, guard_batch, config)
        if finite_tensors(proposal.parameters):
            candidate_loss, candidate_trace = _guard_metrics(proposal.parameters, guard_batch, config)
            accepted, reason = _candidate_is_safe(proposal.parameters, candidate_loss,
                                                  candidate_trace, before_loss, safety)
        else:
            candidate_loss = torch.full_like(before_loss, float("inf"))
            candidate_trace = _guard_metrics(current_parameters, guard_batch, config)[1]
            accepted, reason = False, "nonfinite"

        fallback = False
        rollback = False
        if accepted:
            committed = clone_detached(proposal.parameters)
            after_loss, after_trace = candidate_loss.detach(), candidate_trace.detached()
        else:
            fallback_loss, fallback_trace = _guard_metrics(proposal.base_parameters,
                                                           guard_batch, config)
            fallback_ok, fallback_reason = _candidate_is_safe(
                proposal.base_parameters, fallback_loss, fallback_trace, before_loss, safety)
            if fallback_ok:
                committed = clone_detached(proposal.base_parameters)
                after_loss, after_trace = fallback_loss.detach(), fallback_trace.detached()
                fallback = True
            else:
                committed = clone_detached(current_parameters)
                after_loss = before_loss.detach()
                after_trace = _guard_metrics(current_parameters, guard_batch, config)[1].detached()
                rollback = True
                reason = f"{reason};adam_{fallback_reason}"

        reward = (before_loss.detach().clamp_min(1e-8).log()
                  - after_loss.clamp_min(1e-8).log()).clamp(-1, 1)
        old_feedback = proposal.next_state.feedback
        ema = 0.95 * old_feedback.loss_ema + 0.05 * after_loss
        feedback = Feedback(
            after_loss,
            ema.detach(),
            reward.detach(),
            (old_feedback.loss - after_loss).detach(),
            scalar(1.0 if accepted else 0.0, after_loss),
            scalar(0.0 if accepted else 1.0, after_loss),
            scalar(1.0 if rollback else 0.0, after_loss),
            scalar(progress, after_loss),
        )
        if rollback:
            # A failed policy and failed fallback is a true transaction rollback:
            # parameters, moments, step, and previous updates all remain unchanged.
            previous = proposal.previous_state
            next_state = OptimizerState(previous.momentum, previous.variance,
                                        previous.previous_updates, previous.step, feedback)
        else:
            chosen_updates = (proposal.base_updates if fallback else proposal.raw_updates)
            next_state = OptimizerState(
                proposal.next_state.momentum,
                proposal.next_state.variance,
                tuple(value.detach() for value in chosen_updates),
                proposal.next_state.step,
                feedback,
            )
        return GuardResult(committed, next_state.detached(), accepted, fallback,
                           rollback, reason, before_loss.detach(), after_loss,
                           reward.detach(), after_trace)


def pair_stdp_delta(pre: Tensor, post: Tensor, trace_decay: float = 0.8) -> Tensor:
    """Reference vectorized pair-STDP eligibility for [time,batch,neurons]."""
    if pre.ndim != 3 or post.ndim != 3 or pre.shape[:2] != post.shape[:2]:
        raise ValueError("pre/post must be [time,batch,neurons] with matching time/batch")
    pre_trace = torch.zeros_like(pre[0])
    post_trace = torch.zeros_like(post[0])
    delta = torch.zeros(post.shape[2], pre.shape[2], device=pre.device, dtype=pre.dtype)
    for time_index in range(pre.shape[0]):
        pre_trace = trace_decay * pre_trace + pre[time_index]
        delta = delta + post[time_index].transpose(0, 1) @ pre_trace
        delta = delta - post_trace.transpose(0, 1) @ pre[time_index]
        post_trace = trace_decay * post_trace + post[time_index]
    return delta / (pre.shape[0] * pre.shape[1])


def batch_indices(count: int, batch_size: int, device: torch.device,
                  generator: torch.Generator) -> Iterable[Tensor]:
    order = torch.randperm(count, device=device, generator=generator)
    for start in range(0, count, batch_size):
        yield order[start:start + batch_size]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 4
    inner_lr: float = 3e-3
    stdp_lr: float = 2e-3
    meta_lr: float = 2e-4
    meta_steps: int = 20
    unroll: int = 2
    episode_length: int = 32
    trust_coefficient: float = 0.01
    value_coefficient: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True

    def validate(self) -> None:
        float_values = (self.inner_lr, self.stdp_lr, self.meta_lr,
                        self.trust_coefficient, self.value_coefficient, self.grad_clip)
        if not all(math.isfinite(value) for value in float_values):
            raise ValueError("all training hyperparameters must be finite")
        if min(self.batch_size, self.epochs, self.meta_steps, self.unroll,
               self.episode_length) < 1:
            raise ValueError("batch_size/epochs/meta_steps/unroll/episode_length must be positive")
        if min(self.inner_lr, self.stdp_lr, self.meta_lr, self.grad_clip) <= 0:
            raise ValueError("learning rates and grad_clip must be positive")
        if self.trust_coefficient < 0 or self.value_coefficient < 0:
            raise ValueError("meta-loss coefficients cannot be negative")
        if self.unroll > self.episode_length:
            raise ValueError("unroll cannot exceed the persistent episode length")


def _take_random_batch(split: tuple[Tensor, Tensor], batch_size: int,
                       generator: torch.Generator) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, len(split[0]), (batch_size,), device=split[0].device,
                            generator=generator)
    return split[0][indices], split[1][indices]


def meta_train_policy(
    policy: AlphaZeroPolicyOptimizer,
    snn_config: SNNConfig,
    task_config: TaskConfig,
    train_config: TrainConfig,
    safety: SafetyConfig,
    device: torch.device,
    *,
    seed: int,
    optimizer: torch.optim.Optimizer | None = None,
    retain_final_gradients: bool = False,
) -> tuple[AlphaZeroPolicyOptimizer, torch.optim.Optimizer, list[dict[str, float]]]:
    """First-order truncated meta-learning on disjoint synthetic task seeds."""
    train_config.validate()
    if optimizer is None:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=train_config.meta_lr,
                                      fused=True)
    policy.train()
    gen = cuda_generator(device, seed * 1009 + 7)
    history: list[dict[str, float]] = []
    task: SyntheticTask | None = None
    parameters: tuple[Tensor, ...] | None = None
    state: OptimizerState | None = None
    episode_age = train_config.episode_length
    episode = 0
    for meta_step in range(train_config.meta_steps):
        # Persistent optimizee trajectories expose the policy to young and old
        # optimizer states.  Truncation happens at each outer update, while a
        # fresh task/initialization is used only at the declared episode horizon.
        if task is None or parameters is None or state is None or episode_age >= train_config.episode_length:
            task = make_synthetic_task(task_config, device, seed * 100_003 + episode)
            parameters = initialize_snn(snn_config, device, seed * 65_537 + episode)
            initial_batch = _take_random_batch(task.train, train_config.batch_size, gen)
            initial_loss = loss_only(parameters, initial_batch, snn_config)
            if not bool(torch.isfinite(initial_loss)):
                raise FloatingPointError("non-finite initial meta-training loss")
            state = OptimizerState.zeros(parameters, initial_loss.detach())
            episode_age = 0
            episode += 1
        outer_losses: list[Tensor] = []
        rewards: list[Tensor] = []
        accepted_count = 0

        for inner in range(train_config.unroll):
            support = _take_random_batch(task.train, train_config.batch_size, gen)
            query = _take_random_batch(task.validation, train_config.batch_size, gen)
            _, gradients, trace = loss_and_gradients(parameters, support, snn_config)
            proposal = propose_learned_update(
                policy, parameters, gradients, trace, state, train_config.inner_lr,
                safety, amp=train_config.amp,
                residual_scale=min(1.0, (meta_step + 1) / max(1, train_config.meta_steps // 4)),
            )
            if not finite_tensors(proposal.parameters):
                raise FloatingPointError("policy produced non-finite candidate parameters")
            # Query loss stays differentiable through the bounded policy edit.
            query_result = snn_forward(proposal.parameters, query[0], snn_config,
                                       spike_mode="surrogate", collect_eligibility=False)
            query_loss = F.cross_entropy(query_result.logits, query[1])
            if not bool(torch.isfinite(query_loss.detach())):
                raise FloatingPointError("policy candidate produced a non-finite meta-loss")
            with torch.no_grad():
                before_query, _ = _guard_metrics(parameters, query, snn_config)
                candidate_ok, _ = _candidate_is_safe(
                    proposal.parameters, query_loss.detach(), query_result.trace.detached(),
                    before_query, safety)
                if candidate_ok:
                    committed_loss = query_loss.detach()
                    committed_parameters = tuple(value for value in proposal.parameters)
                    committed_state = proposal.next_state
                    chosen_updates = proposal.raw_updates
                    rolled_back = False
                else:
                    fallback_loss, fallback_trace = _guard_metrics(
                        proposal.base_parameters, query, snn_config)
                    fallback_ok, _ = _candidate_is_safe(
                        proposal.base_parameters, fallback_loss, fallback_trace,
                        before_query, safety)
                    if fallback_ok:
                        committed_loss = fallback_loss.detach()
                        committed_parameters = clone_detached(proposal.base_parameters)
                        committed_state = proposal.next_state
                        chosen_updates = proposal.base_updates
                        rolled_back = False
                    else:
                        committed_loss = before_query.detach()
                        committed_parameters = tuple(value for value in parameters)
                        committed_state = state
                        chosen_updates = state.previous_updates
                        rolled_back = True
                reward = (before_query.clamp_min(1e-8).log()
                          - committed_loss.clamp_min(1e-8).log()).clamp(-1, 1)
            value_loss = F.mse_loss(proposal.predicted_value, reward)
            outer_losses.append(query_loss
                                + train_config.trust_coefficient * proposal.trust_penalty
                                + train_config.value_coefficient * value_loss)
            rewards.append(reward)

            if candidate_ok:
                accepted_count += 1
            parameters = committed_parameters
            feedback = Feedback(
                committed_loss,
                (0.95 * state.feedback.loss_ema + 0.05 * committed_loss).detach(),
                reward,
                (state.feedback.loss - committed_loss).detach(),
                scalar(1.0 if candidate_ok else 0.0, committed_loss),
                scalar(0.0 if candidate_ok else 1.0, committed_loss),
                scalar(1.0 if rolled_back else 0.0, committed_loss),
                scalar((meta_step + 1) / train_config.meta_steps, committed_loss),
            )
            state = OptimizerState(
                committed_state.momentum,
                committed_state.variance,
                tuple(value.detach() for value in chosen_updates),
                committed_state.step,
                feedback,
            )

        meta_loss = torch.stack(outer_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        gradients_finite = finite_tensors(
            parameter.grad for parameter in policy.parameters() if parameter.grad is not None
        )
        if not gradients_finite:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite policy gradient; meta-step aborted")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), train_config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        # Truncate the policy graph but retain the SNN/Adam/feedback trajectory.
        parameters = clone_detached(parameters)
        state = state.detached()
        episode_age += train_config.unroll
        history.append({
            "step": float(meta_step + 1),
            "meta_loss": meta_loss.detach().item(),
            "reward": torch.stack(rewards).mean().item(),
            "accepted_fraction": accepted_count / train_config.unroll,
            "policy_grad_norm": grad_norm.detach().item(),
        })
        if not (retain_final_gradients and meta_step + 1 == train_config.meta_steps):
            optimizer.zero_grad(set_to_none=True)
    return policy, optimizer, history


@dataclass(frozen=True)
class DeploymentResult:
    parameters: tuple[Tensor, ...]
    losses: tuple[float, ...]
    accepted: int
    fallbacks: int
    rollbacks: int
    elapsed_seconds: float
    incremental_peak_vram_bytes: int


def deploy_policy(
    policy: AlphaZeroPolicyOptimizer,
    initial_parameters: Sequence[Tensor],
    train_split: tuple[Tensor, Tensor],
    config: SNNConfig,
    train_config: TrainConfig,
    safety: SafetyConfig,
    *,
    seed: int,
    guarded: bool = True,
) -> DeploymentResult:
    train_config.validate()
    policy.eval()
    parameters = clone_detached(initial_parameters)
    gen = cuda_generator(train_split[0].device, seed)
    guard_gen = cuda_generator(train_split[0].device, seed * 104_729 + 17)
    warmup_batch = _take_random_batch(train_split, train_config.batch_size, gen)
    initial_loss = loss_only(parameters, warmup_batch, config)
    if not bool(torch.isfinite(initial_loss)):
        raise FloatingPointError("non-finite initial deployment loss")
    state = OptimizerState.zeros(parameters, initial_loss.detach())
    losses: list[float] = []
    accepted = fallbacks = rollbacks = 0
    torch.cuda.synchronize(train_split[0].device)
    baseline_memory = torch.cuda.memory_allocated(train_split[0].device)
    torch.cuda.reset_peak_memory_stats(train_split[0].device)
    started = time.perf_counter()
    total_steps = train_config.epochs * math.ceil(len(train_split[0]) / train_config.batch_size)
    taken = 0
    for epoch in range(train_config.epochs):
        epoch_gen = cuda_generator(train_split[0].device, seed * 8191 + epoch)
        for indices in batch_indices(len(train_split[0]), train_config.batch_size,
                                     train_split[0].device, epoch_gen):
            batch = train_split[0][indices], train_split[1][indices]
            _, gradients, trace = loss_and_gradients(parameters, batch, config)
            guard_batch = _take_random_batch(train_split, train_config.batch_size, guard_gen)
            with torch.no_grad():
                proposal = propose_learned_update(
                    policy, parameters, gradients, trace, state, train_config.inner_lr,
                    safety, amp=train_config.amp)
            if guarded:
                outcome = guarded_commit(
                    parameters, proposal, guard_batch, config, safety,
                    progress=(taken + 1) / total_steps)
                parameters, state = outcome.parameters, outcome.state
                accepted += int(outcome.accepted_policy)
                fallbacks += int(outcome.used_adam_fallback)
                rollbacks += int(outcome.rolled_back)
                losses.append(outcome.after_loss.item())
            else:
                if not finite_tensors(proposal.parameters):
                    raise FloatingPointError("raw policy produced non-finite parameters")
                with torch.no_grad():
                    before_loss, _ = _guard_metrics(parameters, guard_batch, config)
                    after_loss, _ = _guard_metrics(proposal.parameters, guard_batch, config)
                    if not torch.isfinite(after_loss):
                        raise FloatingPointError("raw policy produced a non-finite loss")
                    reward = (before_loss.clamp_min(1e-8).log()
                              - after_loss.clamp_min(1e-8).log()).clamp(-1, 1)
                    feedback = Feedback(
                        after_loss.detach(),
                        (0.95 * state.feedback.loss_ema + 0.05 * after_loss).detach(),
                        reward.detach(),
                        (state.feedback.loss - after_loss).detach(),
                        torch.ones_like(after_loss), torch.zeros_like(after_loss),
                        torch.zeros_like(after_loss),
                        scalar((taken + 1) / total_steps, after_loss),
                    )
                parameters = clone_detached(proposal.parameters)
                state = dataclasses.replace(proposal.next_state, feedback=feedback).detached()
                losses.append(after_loss.detach().item())
            taken += 1
    torch.cuda.synchronize(train_split[0].device)
    elapsed = time.perf_counter() - started
    peak = max(0, torch.cuda.max_memory_allocated(train_split[0].device) - baseline_memory)
    return DeploymentResult(parameters, tuple(losses), accepted, fallbacks,
                            rollbacks, elapsed, peak)


def train_surrogate_bptt(
    initial_parameters: Sequence[Tensor],
    train_split: tuple[Tensor, Tensor],
    config: SNNConfig,
    train_config: TrainConfig,
    safety: SafetyConfig,
    *,
    seed: int,
) -> DeploymentResult:
    train_config.validate()
    parameters = clone_detached(initial_parameters)
    dummy_loss = torch.ones((), device=train_split[0].device)
    state = OptimizerState.zeros(parameters, dummy_loss)
    losses: list[float] = []
    torch.cuda.synchronize(train_split[0].device)
    baseline_memory = torch.cuda.memory_allocated(train_split[0].device)
    torch.cuda.reset_peak_memory_stats(train_split[0].device)
    started = time.perf_counter()
    for epoch in range(train_config.epochs):
        gen = cuda_generator(train_split[0].device, seed * 8191 + epoch)
        for indices in batch_indices(len(train_split[0]), train_config.batch_size,
                                     train_split[0].device, gen):
            batch = train_split[0][indices], train_split[1][indices]
            loss, gradients, _ = loss_and_gradients(
                parameters, batch, config, collect_eligibility=False)
            if not finite_tensors((loss, *gradients)):
                raise FloatingPointError("non-finite surrogate-BPTT loss/gradient")
            adam = adam_proposal(gradients, state, train_config.inner_lr)
            updates = tuple(_clip_update(update, parameter, safety)
                            for update, parameter in zip(adam.updates, parameters))
            parameters = _project_parameters(
                tuple(parameter + update for parameter, update in zip(parameters, updates)), safety)
            parameters = clone_detached(parameters)
            feedback = Feedback.initial(loss.detach())
            state = OptimizerState(tuple(v.detach() for v in adam.momentum),
                                   tuple(v.detach() for v in adam.variance),
                                   tuple(v.detach() for v in updates), state.step + 1,
                                   feedback)
            losses.append(loss.detach().item())
    torch.cuda.synchronize(train_split[0].device)
    return DeploymentResult(parameters, tuple(losses), 0, 0, 0,
                            time.perf_counter() - started,
                            max(0, torch.cuda.max_memory_allocated(train_split[0].device)
                                - baseline_memory))


def loss_and_readout_gradients(
    parameters: Sequence[Tensor],
    batch: tuple[Tensor, Tensor],
    config: SNNConfig,
) -> tuple[Tensor, tuple[Tensor, ...], SNNTrace]:
    """Hard SNN/STDP statistics with gradients only for the linear readout."""
    # Hidden parameters are already detached at the step boundary; reuse them
    # instead of cloning the entire SNN merely to differentiate the readout.
    leaves = list(parameters[:-2])
    leaves.extend((parameters[-2].detach().requires_grad_(True),
                   parameters[-1].detach().requires_grad_(True)))
    result = snn_forward(leaves, batch[0], config, spike_mode="hard")
    loss = F.cross_entropy(result.logits, batch[1])
    readout_gradients = torch.autograd.grad(loss, (leaves[-2], leaves[-1]))
    return loss.detach(), tuple(value.detach() for value in readout_gradients), result.trace.detached()


def train_stdp(
    initial_parameters: Sequence[Tensor],
    train_split: tuple[Tensor, Tensor],
    config: SNNConfig,
    train_config: TrainConfig,
    safety: SafetyConfig,
    *,
    seed: int,
) -> DeploymentResult:
    """Pair-STDP hidden layers plus a supervised Adam readout."""
    train_config.validate()
    parameters = clone_detached(initial_parameters)
    dummy_loss = torch.ones((), device=train_split[0].device)
    state = OptimizerState.zeros(parameters[-2:], dummy_loss)
    losses: list[float] = []
    torch.cuda.synchronize(train_split[0].device)
    baseline_memory = torch.cuda.memory_allocated(train_split[0].device)
    torch.cuda.reset_peak_memory_stats(train_split[0].device)
    started = time.perf_counter()
    for epoch in range(train_config.epochs):
        gen = cuda_generator(train_split[0].device, seed * 8191 + epoch)
        for indices in batch_indices(len(train_split[0]), train_config.batch_size,
                                     train_split[0].device, gen):
            batch = train_split[0][indices], train_split[1][indices]
            loss, readout_gradients, trace = loss_and_readout_gradients(parameters, batch, config)
            if not finite_tensors((loss, *readout_gradients, *trace.eligibility)):
                raise FloatingPointError("non-finite STDP/readout statistics")
            adam = adam_proposal(readout_gradients, state, train_config.inner_lr)
            updates: list[Tensor] = []
            final_weight_index = 2 * (config.trainable_layers - 1)
            for index, parameter in enumerate(parameters):
                if index % 2 == 0 and index < final_weight_index:
                    eligibility = trace.eligibility[index // 2]
                    update = train_config.stdp_lr * _matched_basis(eligibility, parameter)
                elif index < final_weight_index:
                    update = torch.zeros_like(parameter)  # hidden biases are local-rule neutral
                else:
                    update = adam.updates[index - final_weight_index]  # supervised readout
                updates.append(_clip_update(update, parameter, safety))
            parameters = _project_parameters(
                tuple(parameter + update for parameter, update in zip(parameters, updates)), safety)
            parameters = clone_detached(parameters)
            state = OptimizerState(tuple(v.detach() for v in adam.momentum),
                                   tuple(v.detach() for v in adam.variance),
                                   tuple(v.detach() for v in updates[-2:]), state.step + 1,
                                   Feedback.initial(loss.detach()))
            losses.append(loss.detach().item())
    torch.cuda.synchronize(train_split[0].device)
    return DeploymentResult(parameters, tuple(losses), 0, 0, 0,
                            time.perf_counter() - started,
                            max(0, torch.cuda.max_memory_allocated(train_split[0].device)
                                - baseline_memory))


def parse_size_spec(spec: str, task: TaskConfig, timesteps: int) -> SNNConfig:
    try:
        depth_text, width_text = spec.lower().split("x", 1)
        depth, width = int(depth_text), int(width_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid size {spec!r}; expected DEPTHxWIDTH") from exc
    if depth < 1 or width < 1:
        raise ValueError("depth and width must be positive")
    config = SNNConfig(task.input_size, (width,) * depth, task.classes, timesteps=timesteps)
    config.validate()
    return config


def source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verification_identity(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema": VERIFY_SCHEMA,
        "source_sha256": source_digest(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unknown")),
        "device_total_memory": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "cudnn": torch.backends.cudnn.version(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "torch_git": torch.version.git_version,
        "policy": dataclasses.asdict(PolicyConfig()),
    }


class VerificationSuite:
    def __init__(self, device: torch.device, *, full_policy: bool = True):
        self.device = device
        self.full_policy = full_policy
        self.results: list[dict[str, object]] = []

    def check(self, name: str, function: Callable[[], None]) -> None:
        torch.cuda.synchronize(self.device)
        started = time.monotonic()
        function()
        torch.cuda.synchronize(self.device)
        self.results.append({"name": name, "passed": True,
                             "seconds": round(time.monotonic() - started, 4)})
        print(f"  PASS {name} ({self.results[-1]['seconds']:.3f}s)", flush=True)

    def run(self) -> list[dict[str, object]]:
        self.check("synthetic_reproducibility_and_balance", self.synthetic_data)
        self.check("synthetic_nonlinearity_controls", self.synthetic_nonlinearity)
        self.check("snn_recurrence_reference", self.recurrence_reference)
        self.check("surrogate_finite_difference", self.surrogate_gradient)
        self.check("pair_stdp_causality", self.stdp_causality)
        self.check("adam_oracle_10_steps", self.adam_oracle)
        self.check("zero_policy_equals_projected_adam_100_steps", self.zero_policy_adam)
        self.check("trust_region_and_nonfinite_rollback", self.safety_properties)
        self.check("tiny_meta_backward_and_update", self.tiny_meta_step)
        self.check("multi_hidden_nonsquare_indexing", self.multi_hidden_indexing)
        self.check("tiny_paired_methods_end_to_end", self.tiny_methods)
        self.check("mixed_precision_policy_parity", self.amp_parity)
        if self.full_policy:
            self.check("full_policy_50m_cuda_forward_backward", self.full_policy_check)
        return self.results

    def synthetic_data(self) -> None:
        config = TaskConfig(input_size=4, classes=2, clusters_per_class=2,
                            train_samples=16, validation_samples=8, test_samples=8)
        first = make_synthetic_task(config, self.device, 3)
        second = make_synthetic_task(config, self.device, 3)
        third = make_synthetic_task(config, self.device, 4)
        if not torch.equal(first.train[0], second.train[0]):
            raise AssertionError("same synthetic seed is not reproducible")
        if torch.equal(first.train[0], third.train[0]):
            raise AssertionError("different synthetic seeds produced the same data")
        for split in (first.train, first.validation, first.test):
            counts = torch.bincount(split[1], minlength=config.classes)
            if int(counts.max() - counts.min()) > 1:
                raise AssertionError("synthetic split is not balanced")
            if split[0].device.type != "cuda" or not torch.isfinite(split[0]).all():
                raise AssertionError("synthetic data left CUDA or became non-finite")

    def recurrence_reference(self) -> None:
        config = SNNConfig(2, (2,), 2, timesteps=2, beta=0.5, threshold=0.4,
                           input_scale=1.0)
        values = (
            torch.tensor([[0.5, -0.2], [0.1, 0.6]], device=self.device),
            torch.tensor([0.0, 0.1], device=self.device),
            torch.tensor([[0.3, -0.4], [0.2, 0.5]], device=self.device),
            torch.tensor([0.05, -0.1], device=self.device),
        )
        inputs = torch.tensor([[1.0, 0.5]], device=self.device)
        actual = snn_forward(values, inputs, config, spike_mode="hard",
                             collect_eligibility=False).logits
        hidden_v = torch.zeros(1, 2, device=self.device)
        previous = torch.zeros_like(hidden_v)
        out_v = torch.zeros(1, 2, device=self.device)
        total = torch.zeros_like(out_v)
        for _ in range(2):
            hidden_v = 0.5 * hidden_v + inputs @ values[0].t() + values[1] - previous * 0.4
            previous = (hidden_v >= 0.4).float()
            out_v = 0.5 * out_v + previous @ values[2].t() + values[3]
            total += out_v
        expected = total / 2
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def synthetic_nonlinearity(self) -> None:
        config = TaskConfig(train_samples=1024, validation_samples=256,
                            test_samples=256)
        task = make_synthetic_task(config, self.device, 900_001)
        controls = task_suitability_check(task, config, steps=100)
        if controls["linear_validation_accuracy"] > controls["chance"] + 0.30:
            raise AssertionError("synthetic control unexpectedly became linearly separable")
        if controls["mlp_validation_accuracy"] < 0.75:
            raise AssertionError("synthetic MLP control failed to learn the task")
        seed_everything(0)

    def surrogate_gradient(self) -> None:
        centered = torch.tensor([-0.7, -0.1, 0.3, 0.9], device=self.device,
                                dtype=torch.float64, requires_grad=True)
        alpha = 2.0
        analytic = torch.autograd.grad(soft_spike(centered, alpha).sum(), centered)[0]
        expected = 1.0 / (1.0 + (alpha * centered).square())
        torch.testing.assert_close(analytic, expected, rtol=1e-12, atol=1e-12)
        eps = 1e-6
        numeric = ((soft_spike(centered.detach() + eps, alpha)
                    - soft_spike(centered.detach() - eps, alpha)) / (2 * eps))
        torch.testing.assert_close(analytic, numeric, rtol=2e-6, atol=2e-8)
        hard = SurrogateSpike.apply(centered, alpha)
        hard_grad = torch.autograd.grad(hard.sum(), centered)[0]
        torch.testing.assert_close(hard_grad, expected, rtol=1e-12, atol=1e-12)

    def stdp_causality(self) -> None:
        causal_pre = torch.zeros(4, 1, 1, device=self.device)
        causal_post = torch.zeros_like(causal_pre)
        causal_pre[0] = 1; causal_post[1] = 1
        anti_pre = torch.zeros_like(causal_pre); anti_post = torch.zeros_like(causal_pre)
        anti_post[0] = 1; anti_pre[1] = 1
        if not bool(pair_stdp_delta(causal_pre, causal_post) > 0):
            raise AssertionError("causal pre-before-post pairing did not potentiate")
        if not bool(pair_stdp_delta(anti_pre, anti_post) < 0):
            raise AssertionError("anti-causal post-before-pre pairing did not depress")
        if not torch.equal(pair_stdp_delta(torch.zeros_like(causal_pre), causal_post),
                           torch.zeros(1, 1, device=self.device)):
            raise AssertionError("quiet presynaptic input changed STDP weights")

    def adam_oracle(self) -> None:
        initial = torch.tensor([0.4, -0.7, 1.2], device=self.device)
        ours = (initial.clone(),)
        loss0 = torch.ones((), device=self.device)
        state = OptimizerState.zeros(ours, loss0)
        reference = nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([reference], lr=0.003)
        for step in range(10):
            gradient = torch.tensor([0.1 + step * 0.01, -0.3, 0.7 - step * 0.02],
                                    device=self.device)
            proposal = adam_proposal((gradient,), state, 0.003)
            ours = (ours[0] + proposal.updates[0],)
            state = OptimizerState(proposal.momentum, proposal.variance,
                                   proposal.updates, state.step + 1, state.feedback)
            reference.grad = gradient.clone()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(ours[0], reference.detach(), rtol=2e-6, atol=2e-7)

    def _tiny_policy(self) -> AlphaZeroPolicyOptimizer:
        with torch.device(self.device):
            return AlphaZeroPolicyOptimizer(
                PolicyConfig(channels=16, blocks=1, groups=4, sketch_bins=8,
                             checkpoint_blocks=False), enforce_floor=False)

    def _tiny_problem(self) -> tuple[SNNConfig, tuple[Tensor, ...], tuple[Tensor, Tensor], Tensor,
                                     tuple[Tensor, ...], SNNTrace, OptimizerState]:
        config = SNNConfig(4, (4,), 2, timesteps=3)
        params = initialize_snn(config, self.device, 11)
        gen = cuda_generator(self.device, 12)
        batch = (torch.rand(8, 4, device=self.device, generator=gen),
                 torch.arange(8, device=self.device) % 2)
        loss, gradients, trace = loss_and_gradients(params, batch, config)
        state = OptimizerState.zeros(params, loss.detach())
        return config, params, batch, loss, gradients, trace, state

    def zero_policy_adam(self) -> None:
        policy = self._tiny_policy().eval()
        config, params, _, _, gradients, trace, state = self._tiny_problem()
        safety = SafetyConfig(max_parameter_update_ratio=1e-4, max_element_update=10.0,
                              max_weight_abs=100.0, max_bias_abs=100.0)
        with torch.no_grad():
            first_adam = adam_proposal(gradients, state, 1.0)
            board, _ = build_observation_board(params, gradients, first_adam, state,
                                               trace, bins=policy.config.sketch_bins)
            if board.shape[-1] != policy.config.sketch_bins:
                raise AssertionError("custom policy sketch width was ignored")
            for step in range(100):
                # Vary the gradients so the moment and bias-correction paths are
                # tested rather than repeating one stationary update.
                step_gradients = tuple(
                    gradient * (1.0 + 0.001 * step) + (step % 3 - 1) * 1e-4
                    for gradient in gradients
                )
                learned = propose_learned_update(policy, params, step_gradients,
                                                 trace, state, 1.0, safety, amp=False)
                for actual, wanted in zip(learned.parameters, learned.base_parameters):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                if learned.trust_penalty.item() != 0.0:
                    raise AssertionError("zero policy incurred a nonzero trust penalty")
                params = clone_detached(learned.parameters)
                state = learned.next_state.detached()

    def safety_properties(self) -> None:
        policy = self._tiny_policy().eval()
        config, params, batch, loss, gradients, trace, state = self._tiny_problem()
        safety = SafetyConfig(max_policy_residual_ratio=0.05,
                              max_parameter_update_ratio=0.01,
                              max_loss_increase_ratio=10.0,
                              max_loss_increase_absolute=10.0,
                              min_spike_rate=0.0)
        with torch.no_grad():
            policy.actor_head[-1].bias.fill_(5.0)
            proposal = propose_learned_update(policy, params, gradients, trace, state,
                                             1e-3, safety, amp=False)
        for index, (update, base_update, parameter) in enumerate(zip(
                proposal.raw_updates, proposal.base_updates, params)):
            if bool(tensor_rms(update) > safety.max_parameter_update_ratio
                    * tensor_rms(parameter).clamp_min(1e-3) + 1e-7):
                raise AssertionError("trust radius was violated")
            if index % 2:
                torch.testing.assert_close(update, base_update, rtol=0, atol=0)
            else:
                ratio = tensor_rms(update - base_update) / tensor_rms(base_update).clamp_min(1e-8)
                if bool(ratio > safety.max_policy_residual_ratio + 5e-5):
                    raise AssertionError("policy residual exceeded its Adam-relative trust radius")

        candidate_poison = list(proposal.parameters)
        candidate_poison[0] = candidate_poison[0].clone()
        candidate_poison[0].view(-1)[0] = float("nan")
        candidate_bad = dataclasses.replace(proposal, parameters=tuple(candidate_poison))
        fallback = guarded_commit(params, candidate_bad, batch, config, safety,
                                  before_loss=loss.detach())
        if not fallback.used_adam_fallback or fallback.rolled_back:
            raise AssertionError("bad policy candidate did not fall back to safe Adam")

        poisoned = list(proposal.parameters)
        poisoned[0] = poisoned[0].clone()
        poisoned[0].view(-1)[0] = float("nan")
        bad = dataclasses.replace(proposal, parameters=tuple(poisoned),
                                  base_parameters=tuple(torch.full_like(p, float("nan")) for p in params))
        outcome = guarded_commit(params, bad, batch, config, safety, before_loss=loss.detach())
        if not outcome.rolled_back:
            raise AssertionError("non-finite policy+fallback did not roll back")
        for actual, wanted in zip(outcome.parameters, params):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        if outcome.state.step != state.step:
            raise AssertionError("rollback advanced the Adam bias-correction step")
        for actual, wanted in zip(outcome.state.momentum, state.momentum):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for actual, wanted in zip(outcome.state.variance, state.variance):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for actual, wanted in zip(outcome.state.previous_updates, state.previous_updates):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

        finite_but_unbounded = list(proposal.parameters)
        finite_but_unbounded[0] = torch.full_like(
            finite_but_unbounded[0], safety.max_weight_abs * 2)
        safe, reason = _candidate_is_safe(finite_but_unbounded, loss.detach(), trace,
                                          loss.detach(), safety)
        if safe or reason != "parameter_bounds":
            raise AssertionError("commit boundary did not reject finite out-of-bound weights")

        dead_trace = dataclasses.replace(
            trace,
            post_rates=(torch.zeros_like(trace.post_rates[0]),) + trace.post_rates[1:],
            mean_spike_rate=scalar(0.25, trace.mean_spike_rate),
        )
        rate_safety = dataclasses.replace(safety, min_spike_rate=0.01)
        safe, reason = _candidate_is_safe(proposal.parameters, loss.detach(), dead_trace,
                                          loss.detach(), rate_safety)
        if safe or reason != "firing_rate":
            raise AssertionError("a dead hidden layer escaped the per-layer firing-rate guard")

    def tiny_meta_step(self) -> None:
        policy = self._tiny_policy().train()
        config, _, _, _, _, _, _ = self._tiny_problem()
        safety = SafetyConfig(min_spike_rate=0.0, max_loss_increase_ratio=10.0,
                              max_loss_increase_absolute=10.0)
        before = policy.actor_head[-1].weight.detach().clone()
        task = TaskConfig(input_size=4, classes=2, clusters_per_class=2,
                          train_samples=32, validation_samples=16, test_samples=16)
        training = TrainConfig(batch_size=8, meta_steps=2, unroll=2,
                               episode_length=4, meta_lr=1e-3, amp=False)
        policy, _, history = meta_train_policy(policy, config, task, training, safety,
                                               self.device, seed=23)
        if len(history) != 2 or not all(
                math.isfinite(value) for row in history for value in row.values()):
            raise AssertionError("truncated meta-training did not finish with finite metrics")
        if torch.equal(before, policy.actor_head[-1].weight.detach()):
            raise AssertionError("tiny meta optimizer did not change the action head")

    def amp_parity(self) -> None:
        policy = self._tiny_policy().eval()
        # Nonzero actor weights make this a real mixed-precision policy check;
        # at the default zero action both paths would trivially equal Adam.
        with torch.no_grad():
            generator = cuda_generator(self.device, 77)
            policy.actor_head[-1].weight.normal_(0.0, 1e-3, generator=generator)
            policy.actor_head[-1].bias.normal_(0.0, 1e-3, generator=generator)
        config, params, _, _, gradients, trace, state = self._tiny_problem()
        safety = SafetyConfig(min_spike_rate=0.0)
        with torch.no_grad():
            fp32 = propose_learned_update(policy, params, gradients, trace, state,
                                         1e-3, safety, amp=False)
            amp = propose_learned_update(policy, params, gradients, trace, state,
                                        1e-3, safety, amp=True)
        for left, right in zip(fp32.parameters, amp.parameters):
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-6)
        torch.testing.assert_close(fp32.predicted_value, amp.predicted_value,
                                   rtol=2e-2, atol=1e-3)

    def multi_hidden_indexing(self) -> None:
        config = SNNConfig(4, (5, 3), 2, timesteps=3)
        parameters = initialize_snn(config, self.device, 35)
        generator = cuda_generator(self.device, 36)
        batch = (torch.rand(8, 4, device=self.device, generator=generator),
                 torch.arange(8, device=self.device) % 2)
        loss, gradients, trace = loss_and_gradients(parameters, batch, config)
        expected_shapes = tuple(weight.shape for weight in weights_of(parameters))
        if tuple(value.shape for value in trace.eligibility) != expected_shapes:
            raise AssertionError("deep eligibility tensors do not align with weight layers")
        without_eligibility = snn_forward(
            parameters, batch[0], config, spike_mode="hard",
            collect_eligibility=False).trace.eligibility
        if len(without_eligibility) != config.trainable_layers or any(
                value.numel() for value in without_eligibility):
            raise AssertionError("disabled eligibility allocated weight-sized placeholders")
        policy = self._tiny_policy().train()
        state = OptimizerState.zeros(parameters, loss.detach())
        proposal = propose_learned_update(
            policy, parameters, gradients, trace, state, 1e-3,
            SafetyConfig(min_spike_rate=0.0), amp=False)
        if len(proposal.parameters) != 2 * config.trainable_layers:
            raise AssertionError("deep policy proposal lost a parameter layer")
        query = snn_forward(proposal.parameters, batch[0], config,
                            spike_mode="surrogate", collect_eligibility=False)
        objective = F.cross_entropy(query.logits, batch[1]) + proposal.predicted_value.square()
        objective.backward()
        if not finite_tensors(p.grad for p in policy.parameters() if p.grad is not None):
            raise AssertionError("deep nonsquare meta backward was non-finite")

    def tiny_methods(self) -> None:
        task_config = TaskConfig(input_size=4, classes=2, clusters_per_class=2,
                                 train_samples=32, validation_samples=16, test_samples=16)
        task = make_synthetic_task(task_config, self.device, 41)
        config = SNNConfig(4, (4,), 2, timesteps=3)
        initial = initialize_snn(config, self.device, 42)
        training = TrainConfig(batch_size=8, epochs=1, meta_steps=1, unroll=1,
                               episode_length=2, amp=False)
        safety = SafetyConfig(min_spike_rate=0.0, max_loss_increase_ratio=10.0,
                              max_loss_increase_absolute=10.0)
        policy = self._tiny_policy().eval()
        methods = (
            deploy_policy(policy, initial, task.train, config, training, safety,
                          seed=43, guarded=True),
            train_surrogate_bptt(initial, task.train, config, training, safety, seed=43),
            train_stdp(initial, task.train, config, training, safety, seed=43),
        )
        expected_steps = math.ceil(len(task.train[0]) / training.batch_size)
        for result in methods:
            if len(result.losses) != expected_steps or not all(math.isfinite(v) for v in result.losses):
                raise AssertionError("a tiny benchmark method produced incomplete/non-finite history")
            if any(value.device.type != "cuda" or not torch.isfinite(value).all()
                   for value in result.parameters):
                raise AssertionError("a tiny benchmark method left CUDA or became non-finite")
            metrics = evaluate_snn(result.parameters, task.test, config, batch_size=8)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise AssertionError("tiny method evaluation was non-finite")
        policy_result = methods[0]
        if policy_result.accepted + policy_result.fallbacks + policy_result.rollbacks != expected_steps:
            raise AssertionError("guard accounting does not cover every deployment step")
        if any(all(torch.equal(after, before) for after, before in zip(result.parameters, initial))
               for result in methods):
            raise AssertionError("a tiny training method was a complete no-op")
        if torch.equal(methods[2].parameters[0], initial[0]):
            raise AssertionError("pair-STDP did not change the hidden weight")
        torch.testing.assert_close(methods[2].parameters[1], initial[1], rtol=0, atol=0)

    def full_policy_check(self) -> None:
        if not torch.cuda.is_bf16_supported():
            raise AssertionError("production AMP requires a BF16-capable CUDA GPU")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.device(self.device):
            policy = AlphaZeroPolicyOptimizer().train()
        if policy.action_parameter_count() < MIN_POLICY_PARAMETERS:
            raise AssertionError("production action path is below 50M parameters")
        if any(parameter.device != self.device for parameter in policy.parameters()):
            raise AssertionError("a production policy parameter is not on the selected GPU")
        if bool(policy.actor_head[-1].weight.detach().count_nonzero()):
            raise AssertionError("production actor is not zero-initialized to exact Adam")
        task_config = TaskConfig(input_size=4, classes=2, clusters_per_class=2,
                                 train_samples=32, validation_samples=16, test_samples=16)
        snn_config = SNNConfig(4, (4,), 2, timesteps=3)
        training = TrainConfig(batch_size=8, meta_steps=2, unroll=1,
                               episode_length=4, amp=True)
        safety = SafetyConfig(min_spike_rate=0.0, max_loss_increase_ratio=10.0,
                              max_loss_increase_absolute=10.0)
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        old_precision = torch.get_float32_matmul_precision()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # Exercise the production fast-numerics path, full observation/action
            # integration, checkpointed backward, fused AdamW state allocation,
            # and a second step that sends gradients through the initially-zero
            # actor head into its first convolution.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            torch.use_deterministic_algorithms(False)
            policy, outer_optimizer, history = meta_train_policy(
            policy, snn_config, task_config, training, safety,
                self.device, seed=101, retain_final_gradients=True)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
            torch.backends.cudnn.benchmark = old_cudnn_benchmark
            torch.set_float32_matmul_precision(old_precision)
            torch.use_deterministic_algorithms(old_deterministic)
        if len(history) != 2 or not all(
                math.isfinite(value) for row in history for value in row.values()):
            raise AssertionError("full production meta steps did not produce finite history")
        missing = [name for name, parameter in policy.named_parameters()
                   if parameter.grad is None]
        if missing:
            raise AssertionError(f"disconnected full-policy parameters: {missing[:3]}")
        if not finite_tensors(parameter.grad for parameter in policy.parameters()):
            raise AssertionError("full policy backward produced non-finite gradients")
        zero_gradients = [name for name, parameter in policy.named_parameters()
                          if not bool(parameter.grad.abs().sum() > 0)]
        if zero_gradients:
            raise AssertionError(f"zero full-policy gradients: {zero_gradients[:3]}")
        selected = (policy.actor_head[0].weight.grad,
                    policy.actor_head[-1].weight.grad,
                    policy.value_mlp[-1].weight.grad,
                    policy.stem[0].weight.grad,
                    policy.blocks[-1].conv2.weight.grad)
        if any(value is None or not bool(value.abs().sum() > 0) for value in selected):
            raise AssertionError("full meta-gradient did not reach every policy/value path")
        optimizer_tensors = [value for values in outer_optimizer.state.values()
                             for value in values.values() if torch.is_tensor(value)]
        if not optimizer_tensors or any(value.device != self.device for value in optimizer_tensors):
            raise AssertionError("fused policy optimizer state is missing or left CUDA")
        if not finite_tensors(optimizer_tensors):
            raise AssertionError("fused policy optimizer state is non-finite")
        peak = torch.cuda.max_memory_allocated(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        if peak > 0.85 * total:
            raise AssertionError(f"full-policy verification used {peak / 2**30:.2f} GiB (>85% VRAM)")
        del policy, outer_optimizer
        torch.cuda.empty_cache()


def run_verification(device: torch.device, manifest_path: Path,
                     *, full_policy: bool = True) -> dict[str, object]:
    print(f"snn_meta_optimizer verification on {torch.cuda.get_device_name(device)}")
    # Never leave an older same-source success available after a failed rerun.
    manifest_path.unlink(missing_ok=True)
    digest_before = source_digest()
    seed_everything(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    suite = VerificationSuite(device, full_policy=full_policy)
    results = suite.run()
    identity = verification_identity(device)
    if identity["source_sha256"] != digest_before:
        raise RuntimeError("source changed while verification was running; rerun verify")
    manifest = {**identity, "passed": True, "checks": results,
                "created_unix": int(time.time())}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"verification passed: {len(results)}/{len(results)} checks")
    print(f"manifest: {manifest_path}")
    return manifest


def require_verification_manifest(device: torch.device, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"verification manifest missing: run `{Path(__file__).name} verify`")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read verification manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("verification manifest root must be an object")
    expected = verification_identity(device)
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("passed") is not True or mismatches:
        detail = ", ".join(mismatches) if mismatches else "failed status"
        raise RuntimeError(f"verification manifest is stale/incompatible ({detail}); rerun verify")
    checks = manifest.get("checks")
    if not isinstance(checks, list) or not all(isinstance(check, dict) for check in checks):
        raise RuntimeError("verification manifest has malformed checks")
    names = [check.get("name") for check in checks]
    if tuple(names) != VERIFICATION_CHECKS:
        raise RuntimeError("verification manifest does not contain the exact canonical check set")
    if not all(check.get("passed") is True for check in checks):
        raise RuntimeError("verification manifest contains a failed or malformed check")
    return manifest


def task_suitability_check(task: SyntheticTask, task_config: TaskConfig,
                           *, steps: int = 100) -> dict[str, float]:
    """GPU-only linear/MLP controls, fitted on train and scored on validation."""
    if steps < 1:
        raise ValueError("task-suitability steps must be positive")
    device = task.train[0].device
    seed_everything(task.seed + 123)

    def fit(model: nn.Module) -> float:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(steps):
            logits = model(task.train[0])
            loss = F.cross_entropy(logits, task.train[1])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            return model(task.validation[0]).argmax(1).eq(task.validation[1]).float().mean().item()

    with torch.device(device):
        linear_model = nn.Linear(task_config.input_size, task_config.classes)
        mlp_model = nn.Sequential(nn.Linear(task_config.input_size, 128), nn.SiLU(),
                                  nn.Linear(128, 128), nn.SiLU(),
                                  nn.Linear(128, task_config.classes))
    linear = fit(linear_model)
    mlp = fit(mlp_model)
    chance = 1.0 / task_config.classes
    if linear > chance + 0.30:
        raise RuntimeError(f"synthetic task is too linear ({linear:.1%} validation accuracy)")
    if mlp < max(0.75, linear + 0.25):
        raise RuntimeError(f"synthetic task lacks a strong nonlinear oracle ({mlp:.1%})")
    return {"chance": chance, "linear_validation_accuracy": linear,
            "mlp_validation_accuracy": mlp}


def save_checkpoint(path: Path, policy: AlphaZeroPolicyOptimizer,
                    optimizer: torch.optim.Optimizer, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source_sha256": source_digest(),
        "policy_config": dataclasses.asdict(policy.config),
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "metadata": metadata,
    }, path)


def benchmark(args: argparse.Namespace, device: torch.device) -> None:
    manifest_path = Path(args.manifest)
    require_verification_manifest(device, manifest_path)

    stop_values = (args.stop_gap, args.stop_rejection_rate, args.residual_ratio)
    if not all(math.isfinite(value) for value in stop_values):
        raise ValueError("all stopping/safety thresholds must be finite")
    if args.stop_patience < 1 or args.stop_gap < 0 or args.oracle_steps < 1:
        raise ValueError("stop-patience/oracle-steps must be positive and stop-gap nonnegative")
    if not 0 <= args.stop_rejection_rate <= 1 or not 0 <= args.residual_ratio <= 1:
        raise ValueError("rejection and residual ratios must be in [0, 1]")

    task_config = TaskConfig(
        input_size=args.input_size, classes=args.classes,
        clusters_per_class=args.clusters, train_samples=args.train_samples,
        validation_samples=args.validation_samples, test_samples=args.test_samples)
    task_config.validate()
    specifications = [value.strip() for value in args.sizes.split(",") if value.strip()]
    if not specifications:
        raise ValueError("--sizes must contain at least one DEPTHxWIDTH entry")
    architectures = [parse_size_spec(spec, task_config, args.timesteps)
                     for spec in specifications]
    counts = [config.parameter_count for config in architectures]
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        raise ValueError("--sizes must be strictly increasing by SNN parameter count")
    safety = SafetyConfig(max_policy_residual_ratio=args.residual_ratio)
    safety.validate()
    train_config = TrainConfig(
        batch_size=args.batch_size, epochs=args.epochs, inner_lr=args.inner_lr,
        stdp_lr=args.stdp_lr, meta_lr=args.meta_lr,
        meta_steps=args.meta_steps_per_size, unroll=args.unroll,
        amp=not args.no_amp)
    train_config.validate()
    if train_config.amp and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 AMP is unavailable on this GPU; use --no-amp")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)

    eval_task = make_synthetic_task(task_config, device, args.eval_task_seed)
    suitability = task_suitability_check(eval_task, task_config, steps=args.oracle_steps)
    print("task controls:", json.dumps(suitability, sort_keys=True), flush=True)

    # Run standalone baselines before allocating the 50M policy/outer optimizer.
    # This prevents policy residency from becoming an artificial BPTT/STDP VRAM
    # bound, and scalarize each result before proceeding to the next method.
    baseline_records: dict[str, dict[str, object]] = {}
    capacity_bound: dict[str, object] | None = None
    successful_specs: list[str] = []
    successful_architectures: list[SNNConfig] = []
    print("precomputing standalone BPTT/STDP baselines", flush=True)
    for size_index, (spec, config) in enumerate(zip(specifications, architectures)):
        print(f"  baseline {spec}: {config.parameter_count:,} parameters", flush=True)
        initial = bptt = stdp = None
        try:
            initialization_seed = args.seed + size_index
            method_seed = args.seed + 10_000 + size_index
            initial = initialize_snn(config, device, initialization_seed)
            untrained_validation = evaluate_snn(initial, eval_task.validation, config)
            untrained_test = evaluate_snn(initial, eval_task.test, config)
            bptt = train_surrogate_bptt(initial, eval_task.train, config, train_config,
                                        safety, seed=method_seed)
            bptt_validation = evaluate_snn(bptt.parameters, eval_task.validation, config)
            bptt_test = evaluate_snn(bptt.parameters, eval_task.test, config)
            bptt_timing = bptt.elapsed_seconds
            bptt_memory = bptt.incremental_peak_vram_bytes
            del bptt, initial
            bptt = initial = None
            torch.cuda.empty_cache()

            initial = initialize_snn(config, device, initialization_seed)
            stdp = train_stdp(initial, eval_task.train, config, train_config,
                              safety, seed=method_seed)
            stdp_validation = evaluate_snn(stdp.parameters, eval_task.validation, config)
            stdp_test = evaluate_snn(stdp.parameters, eval_task.test, config)
            baseline_records[spec] = {
                "validation": {"untrained": untrained_validation, "bptt": bptt_validation,
                               "stdp": stdp_validation},
                "test": {"untrained": untrained_test, "bptt": bptt_test,
                         "stdp": stdp_test},
                "timing": {"bptt": bptt_timing, "stdp": stdp.elapsed_seconds},
                "memory": {"bptt": bptt_memory,
                           "stdp": stdp.incremental_peak_vram_bytes},
            }
            successful_specs.append(spec)
            successful_architectures.append(config)
        except torch.cuda.OutOfMemoryError:
            capacity_bound = {"reached": True, "stage": "standalone_baseline",
                              "failing_size": spec, "reason": "cuda_out_of_memory"}
            print(f"  CUDA capacity bound at baseline {spec}", flush=True)
            break
        finally:
            del initial, bptt, stdp
            torch.cuda.empty_cache()

    specifications = successful_specs
    architectures = successful_architectures
    results: list[dict[str, object]] = []
    policy: AlphaZeroPolicyOptimizer | None = None
    outer_optimizer: torch.optim.Optimizer | None = None
    bound = capacity_bound

    if specifications:
        seed_everything(args.seed)
        try:
            with torch.device(device):
                policy = AlphaZeroPolicyOptimizer()
            outer_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.meta_lr, fused=True)
        except torch.cuda.OutOfMemoryError:
            policy = None
            outer_optimizer = None
            torch.cuda.empty_cache()
            bound = {"reached": True, "stage": "policy_initialization",
                     "failing_size": specifications[0], "reason": "cuda_out_of_memory"}

    consecutive_failures = 0
    if policy is not None and outer_optimizer is not None:
        for size_index, (spec, config) in enumerate(zip(specifications, architectures)):
            print(f"\npolicy size {spec}: {config.parameter_count:,} SNN parameters", flush=True)
            initial = zero_shot = adapted = unguarded = None
            try:
                initialization_seed = args.seed + size_index
                method_seed = args.seed + 10_000 + size_index
                initial = initialize_snn(config, device, initialization_seed)

                # Zero-shot is measured before the policy sees this new size.
                zero_shot = deploy_policy(policy, initial, eval_task.train, config,
                                          train_config, safety, seed=method_seed)
                zero_validation = evaluate_snn(
                    zero_shot.parameters, eval_task.validation, config)
                zero_test = evaluate_snn(zero_shot.parameters, eval_task.test, config)
                zero_timing = zero_shot.elapsed_seconds
                del zero_shot
                zero_shot = None
                torch.cuda.empty_cache()

                torch.cuda.synchronize(device)
                meta_baseline_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                meta_started = time.perf_counter()
                policy, outer_optimizer, meta_history = meta_train_policy(
                    policy, config, task_config, train_config, safety, device,
                    seed=args.meta_task_seed + size_index * 10_000,
                    optimizer=outer_optimizer)
                torch.cuda.synchronize(device)
                meta_elapsed = time.perf_counter() - meta_started
                meta_memory = max(
                    0, torch.cuda.max_memory_allocated(device) - meta_baseline_memory)

                adapted = deploy_policy(policy, initial, eval_task.train, config,
                                        train_config, safety, seed=method_seed)
                guarded_validation = evaluate_snn(
                    adapted.parameters, eval_task.validation, config)
                guarded_test = evaluate_snn(adapted.parameters, eval_task.test, config)
                guarded_timing = adapted.elapsed_seconds
                guarded_memory = adapted.incremental_peak_vram_bytes
                rejection_rate = ((adapted.fallbacks + adapted.rollbacks)
                                  / max(1, adapted.accepted + adapted.fallbacks
                                        + adapted.rollbacks))
                del adapted
                adapted = None
                torch.cuda.empty_cache()

                unguarded = deploy_policy(policy, initial, eval_task.train, config,
                                          train_config, safety, seed=method_seed,
                                          guarded=False)
                unguarded_validation = evaluate_snn(
                    unguarded.parameters, eval_task.validation, config)
                unguarded_test = evaluate_snn(
                    unguarded.parameters, eval_task.test, config)
                unguarded_timing = unguarded.elapsed_seconds
                unguarded_memory = unguarded.incremental_peak_vram_bytes

                baseline = baseline_records[spec]
                validation_metrics = dict(baseline["validation"])
                validation_metrics.update({
                    "policy_zero_shot": zero_validation,
                    "policy_adapted_guarded": guarded_validation,
                    "policy_adapted_unguarded": unguarded_validation,
                })
                test_metrics = dict(baseline["test"])
                test_metrics.update({
                    "policy_zero_shot": zero_test,
                    "policy_adapted_guarded": guarded_test,
                    "policy_adapted_unguarded": unguarded_test,
                })

                def gap(metrics: dict[str, dict[str, float]], method: str) -> float | None:
                    denominator = (metrics["bptt"]["accuracy"]
                                   - metrics["untrained"]["accuracy"])
                    if denominator <= 1e-8:
                        return None
                    return ((metrics["bptt"]["accuracy"] - metrics[method]["accuracy"])
                            / denominator)

                validation_unguarded_gap = gap(
                    validation_metrics, "policy_adapted_unguarded")
                row = {
                    "size": spec,
                    "snn_parameters": config.parameter_count,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "validation_normalized_gap_unguarded": validation_unguarded_gap,
                    "validation_normalized_gap_guarded": gap(
                        validation_metrics, "policy_adapted_guarded"),
                    "test_normalized_gap_unguarded": gap(
                        test_metrics, "policy_adapted_unguarded"),
                    "test_normalized_gap_guarded": gap(
                        test_metrics, "policy_adapted_guarded"),
                    "policy_rejection_rate": rejection_rate,
                    "timing_seconds": {
                        "meta_training": meta_elapsed,
                        "zero_shot": zero_timing,
                        **baseline["timing"],
                        "policy_guarded": guarded_timing,
                        "policy_unguarded": unguarded_timing,
                    },
                    "incremental_peak_vram_bytes": {
                        "meta_training": meta_memory,
                        **baseline["memory"],
                        "policy_guarded": guarded_memory,
                        "policy_unguarded": unguarded_memory,
                    },
                    "last_meta_metrics": meta_history[-1] if meta_history else {},
                }
                results.append(row)
                print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
                save_checkpoint(
                    Path(args.output).with_suffix(f".{spec}.pt"), policy,
                    outer_optimizer,
                    {"completed_sizes": specifications[:size_index + 1],
                     "results": results})

                # Stop on validation only; test is reporting-only.  The quality
                # comparison uses unguarded policy to avoid a guard oracle that
                # BPTT does not receive.  Guard rejection remains a safety bound.
                failed = ((validation_unguarded_gap is not None
                           and validation_unguarded_gap > args.stop_gap)
                          or rejection_rate > args.stop_rejection_rate)
                consecutive_failures = consecutive_failures + 1 if failed else 0
                if consecutive_failures >= args.stop_patience:
                    reasons = []
                    if (validation_unguarded_gap is not None
                            and validation_unguarded_gap > args.stop_gap):
                        reasons.append("validation_optimization_gap")
                    if rejection_rate > args.stop_rejection_rate:
                        reasons.append("safety_rejection_rate")
                    bound = {"reached": True, "stage": "policy_quality",
                             "failing_size": spec, "reason": "+".join(reasons)}
                    print(f"progressive policy bound reached at {spec}", flush=True)
                    break
            except torch.cuda.OutOfMemoryError:
                outer_optimizer.zero_grad(set_to_none=True)
                bound = {"reached": True, "stage": "policy_optimizee",
                         "failing_size": spec, "reason": "cuda_out_of_memory"}
                print(f"CUDA policy/optimizee capacity bound at {spec}", flush=True)
                torch.cuda.empty_cache()
                break
            finally:
                del initial, zero_shot, adapted, unguarded
                torch.cuda.empty_cache()

    if bound is None:
        if results:
            bound = {"reached": False,
                     "lower_bound_size": results[-1]["size"],
                     "lower_bound_snn_parameters": results[-1]["snn_parameters"],
                     "reason": "configured_ladder_exhausted"}
        else:
            bound = {"reached": True, "stage": "setup", "reason": "no_size_completed"}

    output = {
        "source_sha256": source_digest(),
        "device": torch.cuda.get_device_name(device),
        "task": dataclasses.asdict(task_config),
        "task_controls": suitability,
        "policy_parameters": policy.parameter_count() if policy is not None else None,
        "action_parameters": policy.action_parameter_count() if policy is not None else None,
        "bound": bound,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(f"benchmark results: {output_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="run the mandatory GPU verification gate")
    verify.add_argument("--device", dest="device", default=argparse.SUPPRESS,
                        help="CUDA device (accepted here or before the subcommand)")
    verify.add_argument("--manifest", default="build/snn_meta_verification.json")
    verify.add_argument("--tiny-only", action="store_true",
                        help="developer diagnostic only; does not create a benchmark-valid manifest")

    run = subparsers.add_parser("benchmark", help="run the verified progressive benchmark")
    run.add_argument("--device", dest="device", default=argparse.SUPPRESS,
                     help="CUDA device (accepted here or before the subcommand)")
    run.add_argument("--manifest", default="build/snn_meta_verification.json")
    run.add_argument("--output", default="build/snn_meta_results.json")
    run.add_argument(
        "--sizes",
        default=("1x4,1x8,1x16,1x32,1x64,1x128,1x256,1x512,"
                 "2x128,1x1024,3x128,1x2048,4x128,6x128,1x4096,"
                 "8x128,1x8192,12x128,16x128"),
        help="strictly parameter-increasing DEPTHxWIDTH ladder",
    )
    run.add_argument("--input-size", type=int, default=16)
    run.add_argument("--classes", type=int, default=4)
    run.add_argument("--clusters", type=int, default=8)
    run.add_argument("--train-samples", type=int, default=2048)
    run.add_argument("--validation-samples", type=int, default=512)
    run.add_argument("--test-samples", type=int, default=512)
    run.add_argument("--timesteps", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--epochs", type=int, default=4)
    run.add_argument("--inner-lr", type=float, default=3e-3)
    run.add_argument("--stdp-lr", type=float, default=2e-3)
    run.add_argument("--meta-lr", type=float, default=2e-4)
    run.add_argument("--meta-steps-per-size", type=int, default=20)
    run.add_argument("--unroll", type=int, default=2)
    run.add_argument("--residual-ratio", type=float, default=0.10)
    run.add_argument("--oracle-steps", type=int, default=100)
    run.add_argument("--eval-task-seed", type=int, default=900_001)
    run.add_argument("--meta-task-seed", type=int, default=100_001)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--stop-gap", type=float, default=0.50)
    run.add_argument("--stop-rejection-rate", type=float, default=0.25)
    run.add_argument("--stop-patience", type=int, default=2)
    run.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        device = require_cuda(args.device)
        if args.command == "verify":
            if args.tiny_only:
                # Useful while editing, but intentionally cannot unlock experiments.
                seed_everything(0)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                suite = VerificationSuite(device, full_policy=False)
                suite.run()
                print("tiny verification passed; full verify is still required")
            else:
                run_verification(device, Path(args.manifest), full_policy=True)
        else:
            benchmark(args, device)
    except (AssertionError, FloatingPointError, OSError, RuntimeError, ValueError) as exc:
        print(f"snn_meta_optimizer: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
