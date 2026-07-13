#!/usr/bin/env python3
"""Resumable production experiment for the 50M SNN meta-optimizer.

This runner deliberately differs from ``snn_meta_optimizer.py benchmark``:
one policy is meta-trained over a declared architecture distribution, selected
on held-out development tasks, frozen, and only then evaluated on independent
width/depth ladders.  Every durable transition is checksummed and resumable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F
from torch import Tensor

import snn_meta_optimizer as core
from snn_production_support import (
    CheckpointManager,
    EventLogger,
    RunLock,
    SignalController,
    atomic_write_json,
    gpu_telemetry,
    stable_config_digest,
    utc_timestamp,
)


PRODUCTION_SCHEMA = 1
PRODUCTION_VERIFY_SCHEMA = 1
MIB = 2**20
GIB = 2**30


class GracefulStop(RuntimeError):
    pass


class HealthAbort(RuntimeError):
    pass


def production_source_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(core.__file__), Path(__file__).with_name("snn_production_support.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pack_feedback(value: core.Feedback) -> dict[str, Tensor]:
    return {field.name: getattr(value, field.name).detach()
            for field in dataclasses.fields(core.Feedback)}


def unpack_feedback(value: dict[str, Tensor]) -> core.Feedback:
    return core.Feedback(**value)


def pack_optimizer_state(value: core.OptimizerState) -> dict[str, Any]:
    return {
        "momentum": tuple(item.detach() for item in value.momentum),
        "variance": tuple(item.detach() for item in value.variance),
        "previous_updates": tuple(item.detach() for item in value.previous_updates),
        "step": value.step,
        "feedback": pack_feedback(value.feedback),
    }


def unpack_optimizer_state(value: dict[str, Any]) -> core.OptimizerState:
    return core.OptimizerState(
        tuple(value["momentum"]), tuple(value["variance"]),
        tuple(value["previous_updates"]), int(value["step"]),
        unpack_feedback(value["feedback"]),
    )


def finite_scalar(value: Tensor, name: str) -> float:
    detached = value.detach().float()
    if not bool(torch.isfinite(detached)):
        raise FloatingPointError(f"non-finite {name}")
    return detached.item()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


_T_CRITICAL_95 = (
    math.nan, 12.706204736, 4.302652730, 3.182446305, 2.776445105,
    2.570581836, 2.446911851, 2.364624252, 2.306004135, 2.262157163,
    2.228138852, 2.200985160, 2.178812830, 2.160368656, 2.144786688,
    2.131449546, 2.119905299, 2.109815578, 2.100922040, 2.093024054,
    2.085963447, 2.079613845, 2.073873068, 2.068657610, 2.063898562,
    2.059538553, 2.055529439, 2.051830516, 2.048407142, 2.045229642,
    2.042272456,
)


def t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical value without a SciPy dependency."""
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if degrees_of_freedom < len(_T_CRITICAL_95):
        return _T_CRITICAL_95[degrees_of_freedom]
    # Cornish-Fisher expansion around N(0, 1); accurate in the df > 30
    # region where the exact table above ends.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (z + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z)
            / (384 * df**3))


def mean_ci95(values: Sequence[float | None]) -> dict[str, float | int | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    average = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": average, "std": None, "ci95": None}
    std = statistics.stdev(values)
    return {"n": len(values), "mean": average, "std": std,
            "ci95": (t_critical_95(len(values) - 1) * std
                     / math.sqrt(len(values)))}


def alarm_summary(alarms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for alarm in alarms:
        kind = str(alarm.get("type", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "total": len(alarms),
        "by_type": dict(sorted(counts.items())),
        "latest_type": (str(alarms[-1].get("type", "unknown"))
                        if alarms else None),
    }


def reconcile_terminal_status(
    state: dict[str, Any], run_id: str, status_path: Path,
) -> None:
    """Make a lightweight durable error marker authoritative over a checkpoint."""
    if not status_path.is_file():
        return
    latest_status = json.loads(status_path.read_text())
    if (latest_status.get("run_id") == run_id
            and latest_status.get("status") == "error"):
        state["status"] = "error"
        state.setdefault("terminal_error", latest_status.get("data"))


def paired_metric_summary(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]],
    split: str, metric: str,
) -> dict[str, Any]:
    left_by_seed = {int(row["task_seed"]): row for row in left}
    right_by_seed = {int(row["task_seed"]): row for row in right}
    if len(left_by_seed) != len(left) or len(right_by_seed) != len(right):
        raise ValueError("paired comparison contains duplicate task seeds")
    if left_by_seed.keys() != right_by_seed.keys():
        raise ValueError("paired comparison task seeds do not match")
    seeds = sorted(left_by_seed)
    deltas = [
        float(left_by_seed[seed][split][metric])
        - float(right_by_seed[seed][split][metric])
        for seed in seeds
    ]
    result = mean_ci95(deltas)
    result["lower"] = (None if result["ci95"] is None
                       else float(result["mean"]) - float(result["ci95"]))
    result["upper"] = (None if result["ci95"] is None
                       else float(result["mean"]) + float(result["ci95"]))
    result["wins"] = sum(
        delta < 0 if metric == "loss" else delta > 0 for delta in deltas)
    result["task_seeds"] = seeds
    return result


@dataclass(frozen=True)
class ProductionConfig:
    input_size: int = 16
    classes: int = 4
    clusters: int = 8
    train_samples: int = 4096
    validation_samples: int = 2048
    test_samples: int = 4096
    batch_size: int = 64
    timesteps: int = 8
    deployment_steps: int = 256
    meta_max_steps: int = 3000
    meta_train_specs: tuple[str, ...] = (
        "1x64", "1x256", "2x128", "4x128", "2x256", "4x256")
    width_specs: tuple[str, ...] = (
        "2x32", "2x64", "2x128", "2x256", "2x512")
    depth_specs: tuple[str, ...] = (
        "1x128", "2x128", "4x128", "8x128")
    dev_task_seeds: tuple[int, ...] = (700_001, 700_002, 700_003)
    eval_task_seeds: tuple[int, ...] = (900_001, 900_002, 900_003)
    bptt_lrs: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    stdp_lrs: tuple[float, ...] = (5e-4, 2e-3, 8e-3)
    tune_steps: int = 128
    dev_every: int = 50
    dev_steps: int = 256
    meta_lr: float = 2e-4
    terminal_weight: float = 0.75
    trust_coefficient: float = 0.01
    value_coefficient: float = 0.1
    grad_clip: float = 1.0
    residual_ratio: float = 0.10
    residual_warmup: int = 100
    meta_seed: int = 100_001
    policy_seed: int = 1
    max_vram_fraction: float = 0.90
    stop_gap: float = 0.50
    stop_rejection_rate: float = 0.25
    stop_patience: int = 2
    learnability_improvement: float = 0.05
    meta_rollback_window: int = 50
    meta_rollback_warmup_steps: int = 25
    meta_max_rollback_fraction: float = 0.10
    meta_max_consecutive_rollback_steps: int = 3

    def validate(self) -> None:
        ints = (self.input_size, self.classes, self.clusters, self.train_samples,
                self.validation_samples, self.test_samples, self.batch_size,
                self.timesteps, self.deployment_steps, self.meta_max_steps,
                self.tune_steps, self.dev_every, self.dev_steps,
                self.residual_warmup, self.stop_patience,
                self.meta_rollback_window, self.meta_rollback_warmup_steps,
                self.meta_max_consecutive_rollback_steps)
        if min(ints) < 1:
            raise ValueError("all production counts must be positive")
        floats = (*self.bptt_lrs, *self.stdp_lrs, self.meta_lr, self.terminal_weight,
                  self.trust_coefficient, self.value_coefficient, self.grad_clip,
                  self.residual_ratio, self.max_vram_fraction,
                  self.stop_gap, self.stop_rejection_rate,
                  self.learnability_improvement,
                  self.meta_max_rollback_fraction)
        if not all(math.isfinite(item) for item in floats):
            raise ValueError("all production floats must be finite")
        if not 0 <= self.terminal_weight <= 1:
            raise ValueError("terminal_weight must be in [0,1]")
        if not 0 < self.max_vram_fraction < 1:
            raise ValueError("max_vram_fraction must be in (0,1)")
        if not 0 <= self.residual_ratio <= 1:
            raise ValueError("residual_ratio must be in [0,1]")
        if not 0 <= self.learnability_improvement < 1:
            raise ValueError("learnability_improvement must be in [0,1)")
        if not 0 <= self.meta_max_rollback_fraction <= 1:
            raise ValueError("meta_max_rollback_fraction must be in [0,1]")
        task = self.task_config()
        task.validate()
        for spec in (*self.meta_train_specs, *self.width_specs, *self.depth_specs):
            core.parse_size_spec(spec, task, self.timesteps).validate()

    def task_config(self) -> core.TaskConfig:
        return core.TaskConfig(
            input_size=self.input_size, classes=self.classes,
            clusters_per_class=self.clusters, train_samples=self.train_samples,
            validation_samples=self.validation_samples, test_samples=self.test_samples)

    def snn(self, spec: str) -> core.SNNConfig:
        return core.parse_size_spec(spec, self.task_config(), self.timesteps)

    def horizon(self, meta_step: int, peak_bytes: int = 0) -> int:
        if meta_step < 250:
            return 2
        if meta_step < 750:
            return 4
        if meta_step < 2250:
            return 8
        # K=16 is allowed only with at least 2.5 GiB headroom under the declared
        # production ceiling.  Otherwise K=8 is the recorded safe fallback.
        limit = int(self.max_vram_fraction * torch.cuda.get_device_properties(0).total_memory)
        return 16 if peak_bytes < limit - int(2.5 * GIB) else 8


def config_from_args(args: argparse.Namespace) -> ProductionConfig:
    def specs(value: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in value.split(",") if item.strip())

    def ints(value: str) -> tuple[int, ...]:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())

    config = ProductionConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        deployment_steps=args.deployment_steps,
        meta_max_steps=args.meta_max_steps,
        meta_train_specs=specs(args.meta_train_specs),
        width_specs=specs(args.width_specs),
        depth_specs=specs(args.depth_specs),
        dev_task_seeds=ints(args.dev_task_seeds),
        eval_task_seeds=ints(args.eval_task_seeds),
        dev_every=args.dev_every,
        dev_steps=args.dev_steps,
        tune_steps=args.tune_steps,
        meta_seed=args.meta_seed,
        policy_seed=args.policy_seed,
        meta_lr=args.meta_lr,
        learnability_improvement=args.learnability_improvement,
    )
    config.validate()
    return config


def initial_run_state(run_id: str, config: ProductionConfig) -> dict[str, Any]:
    return {
        "schema": PRODUCTION_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "phase": "task_controls",
        "created_utc": utc_timestamp(),
        "meta_step": 0,
        "meta_slots": [None] * len(config.meta_train_specs),
        "meta_history": [],
        "tuning": {},
        "selected_lrs": {},
        "best_dev_loss": None,
        "best_policy_milestone": None,
        "best_meta_step": None,
        "zero_shot_policy_milestone": None,
        "dev_history": [],
        "evaluation_results": [],
        "current_method": None,
        "bound": None,
        "alarms": [],
        "training_termination": None,
        "early_stop": None,
    }


def initialize_slot(config: ProductionConfig, slot_index: int, episode: int,
                    device: torch.device, inner_lr: float) -> dict[str, Any]:
    spec = config.meta_train_specs[slot_index]
    snn = config.snn(spec)
    task_seed = config.meta_seed * 1_000_003 + slot_index * 10_007 + episode
    init_seed = config.meta_seed * 65_537 + slot_index * 1009 + episode
    task = core.make_synthetic_task(config.task_config(), device, task_seed)
    parameters = core.initialize_snn(snn, device, init_seed)
    generator = core.cuda_generator(device, task_seed + 71)
    batch = core._take_random_batch(task.train, config.batch_size, generator)
    loss = core.loss_only(parameters, batch, snn)
    state = core.OptimizerState.zeros(parameters, loss)
    return {
        "spec": spec,
        "episode": episode,
        "age": 0,
        "task_seed": task_seed,
        "init_seed": init_seed,
        "parameters": tuple(item.detach() for item in parameters),
        "optimizer_state": pack_optimizer_state(state),
        "inner_lr": inner_lr,
    }


def _actual_meta_commit(
    parameters: Sequence[Tensor], proposal: core.LearnedProposal,
    query: tuple[Tensor, Tensor], snn: core.SNNConfig,
    safety: core.SafetyConfig,
) -> tuple[
    tuple[Tensor, ...], core.OptimizerState, Tensor, Tensor, Tensor, str,
    core.SNNTrace,
]:
    with torch.no_grad():
        before_loss, before_trace = core._guard_metrics(parameters, query, snn)
        candidate_loss, candidate_trace = core._guard_metrics(
            proposal.parameters, query, snn)
        candidate_ok, candidate_reason = core._candidate_is_safe(
            proposal.parameters, candidate_loss, candidate_trace,
            before_loss, safety)
        if candidate_ok:
            committed_parameters = tuple(proposal.parameters)
            committed_loss = candidate_loss.detach()
            committed_trace = candidate_trace.detached()
            chosen = proposal.raw_updates
            next_state = proposal.next_state
            decision = "accepted"
        else:
            fallback_loss, fallback_trace = core._guard_metrics(
                proposal.base_parameters, query, snn)
            fallback_ok, fallback_reason = core._candidate_is_safe(
                proposal.base_parameters, fallback_loss, fallback_trace,
                before_loss, safety)
            if fallback_ok:
                committed_parameters = core.clone_detached(proposal.base_parameters)
                committed_loss = fallback_loss.detach()
                committed_trace = fallback_trace.detached()
                chosen = proposal.base_updates
                next_state = proposal.next_state
                decision = f"adam_fallback:{candidate_reason}"
            else:
                committed_parameters = tuple(parameters)
                committed_loss = before_loss.detach()
                committed_trace = before_trace.detached()
                chosen = proposal.previous_state.previous_updates
                next_state = proposal.previous_state
                decision = f"rollback:{candidate_reason}+adam_{fallback_reason}"
        reward = (before_loss.clamp_min(1e-8).log()
                  - committed_loss.clamp_min(1e-8).log()).clamp(-1, 1)
        feedback = core.Feedback(
            committed_loss,
            (0.95 * proposal.previous_state.feedback.loss_ema
             + 0.05 * committed_loss).detach(),
            reward.detach(),
            (proposal.previous_state.feedback.loss - committed_loss).detach(),
            core.scalar(1.0 if decision == "accepted" else 0.0, committed_loss),
            core.scalar(0.0 if decision == "accepted" else 1.0, committed_loss),
            core.scalar(1.0 if decision.startswith("rollback") else 0.0, committed_loss),
            proposal.previous_state.feedback.progress,
        )
        state = core.OptimizerState(
            next_state.momentum, next_state.variance,
            tuple(item.detach() for item in chosen), next_state.step, feedback)
    return (committed_parameters, state, before_loss.detach(), committed_loss,
            reward.detach(), decision, committed_trace)


def hidden_layer_rate_stats(trace: core.SNNTrace) -> tuple[Tensor, Tensor]:
    """Return the least/most active hidden-layer rates for safety telemetry."""
    rates = torch.stack([value.detach().mean() for value in trace.post_rates[:-1]])
    return rates.amin(), rates.amax()


def safe_quality_loss(candidate_loss: Tensor, committed_loss: Tensor,
                      decision: str) -> Tensor:
    """Expose quality gradients only for policy actions that were committed."""
    return candidate_loss if decision == "accepted" else committed_loss.detach()


def meta_outer_step(
    policy: core.AlphaZeroPolicyOptimizer,
    optimizer: torch.optim.Optimizer,
    slot: dict[str, Any],
    meta_step: int,
    config: ProductionConfig,
    device: torch.device,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    snn = config.snn(slot["spec"])
    task = core.make_synthetic_task(config.task_config(), device, int(slot["task_seed"]))
    parameters = tuple(slot["parameters"])
    state = unpack_optimizer_state(slot["optimizer_state"])
    safety = core.SafetyConfig(max_policy_residual_ratio=config.residual_ratio)
    generator = core.cuda_generator(
        device, config.meta_seed * 10_000_019 + meta_step * 101 + int(slot["episode"]))
    candidate_losses: list[Tensor] = []
    committed_losses: list[Tensor] = []
    quality_losses: list[Tensor] = []
    trust_losses: list[Tensor] = []
    value_losses: list[Tensor] = []
    rewards: list[Tensor] = []
    support_losses: list[Tensor] = []
    residual_ratios: list[Tensor] = []
    decisions: list[str] = []
    firing_rates: list[Tensor] = []
    min_layer_firing_rates: list[Tensor] = []
    max_layer_firing_rates: list[Tensor] = []
    voltages: list[Tensor] = []

    policy.train()
    for inner in range(horizon):
        support = core._take_random_batch(task.train, config.batch_size, generator)
        query = core._take_random_batch(task.validation, config.batch_size, generator)
        support_loss, gradients, trace = core.loss_and_gradients(parameters, support, snn)
        residual_scale = min(1.0, (meta_step + 1) / config.residual_warmup)
        proposal = core.propose_learned_update(
            policy, parameters, gradients, trace, state, float(slot["inner_lr"]),
            safety, amp=True, residual_scale=residual_scale)
        result = core.snn_forward(proposal.parameters, query[0], snn,
                                  spike_mode="surrogate", collect_eligibility=False)
        candidate_loss = F.cross_entropy(result.logits, query[1])
        if not bool(torch.isfinite(candidate_loss.detach())):
            raise FloatingPointError("non-finite production meta candidate loss")
        (parameters, state, _, committed_loss, reward, decision,
         committed_trace) = _actual_meta_commit(
            parameters, proposal, query, snn, safety)
        state = dataclasses.replace(
            state,
            feedback=dataclasses.replace(
                state.feedback,
                progress=core.scalar(
                    (meta_step + 1) / config.meta_max_steps, state.feedback.loss)))
        value_loss = F.mse_loss(proposal.predicted_value, reward)
        candidate_losses.append(candidate_loss)
        committed_losses.append(committed_loss)
        # Rejected actions must not receive a meta-gradient through an unsafe
        # candidate.  Their quality term is the actually committed fallback or
        # rollback loss (constant with respect to the policy); trust/value terms
        # still train the policy away from the rejection.
        quality_losses.append(safe_quality_loss(
            candidate_loss, committed_loss, decision))
        trust_losses.append(proposal.trust_penalty)
        value_losses.append(value_loss)
        rewards.append(reward)
        support_losses.append(support_loss.detach())
        decisions.append(decision)
        firing_rates.append(committed_trace.mean_spike_rate.detach())
        layer_min, layer_max = hidden_layer_rate_stats(committed_trace)
        min_layer_firing_rates.append(layer_min)
        max_layer_firing_rates.append(layer_max)
        voltages.append(committed_trace.max_abs_voltage.detach())
        for index, (learned, base) in enumerate(zip(
                proposal.raw_updates, proposal.base_updates)):
            if index % 2 == 0:
                residual_ratios.append(
                    core.tensor_rms(learned - base) / core.tensor_rms(base).clamp_min(1e-8))

    quality = (config.terminal_weight * quality_losses[-1]
               + (1.0 - config.terminal_weight) * torch.stack(quality_losses).mean())
    meta_loss = (quality
                 + config.trust_coefficient * torch.stack(trust_losses).mean()
                 + config.value_coefficient * torch.stack(value_losses).mean())
    optimizer.zero_grad(set_to_none=True)
    meta_loss.backward()
    if not core.finite_tensors(
            parameter.grad for parameter in policy.parameters() if parameter.grad is not None):
        raise FloatingPointError("non-finite production policy gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    slot = dict(slot)
    slot["parameters"] = core.clone_detached(parameters)
    slot["optimizer_state"] = pack_optimizer_state(state.detached())
    slot["age"] = int(slot["age"]) + horizon
    fallback_count = sum(item.startswith("adam_fallback") for item in decisions)
    rollback_count = sum(item.startswith("rollback") for item in decisions)
    decision_counts = {
        decision: decisions.count(decision) for decision in sorted(set(decisions))}
    row: dict[str, Any] = {
        "meta_step": meta_step + 1,
        "spec": str(slot["spec"]),
        "horizon": horizon,
        "episode": int(slot["episode"]),
        "episode_age": int(slot["age"]),
        "meta_loss": finite_scalar(meta_loss, "meta_loss"),
        "terminal_query_loss": finite_scalar(candidate_losses[-1], "terminal_query_loss"),
        "mean_query_loss": finite_scalar(torch.stack(candidate_losses).mean(), "query_loss"),
        "terminal_committed_loss": finite_scalar(
            committed_losses[-1], "terminal_committed_loss"),
        "mean_committed_loss": finite_scalar(
            torch.stack(committed_losses).mean(), "committed_loss"),
        "mean_support_loss": finite_scalar(torch.stack(support_losses).mean(), "support_loss"),
        "mean_reward": finite_scalar(torch.stack(rewards).mean(), "reward"),
        "value_loss": finite_scalar(torch.stack(value_losses).mean(), "value_loss"),
        "trust_penalty": finite_scalar(torch.stack(trust_losses).mean(), "trust_penalty"),
        "policy_grad_norm": finite_scalar(grad_norm, "policy_grad_norm"),
        "residual_ratio_mean": finite_scalar(
            torch.stack(residual_ratios).mean(), "residual_ratio") if residual_ratios else 0.0,
        "residual_ratio_max": finite_scalar(
            torch.stack(residual_ratios).amax(), "residual_ratio_max") if residual_ratios else 0.0,
        "accepted_fraction": (horizon - fallback_count - rollback_count) / horizon,
        "fallback_count": fallback_count,
        "rollback_count": rollback_count,
        "decisions": list(decisions),
        "decision_counts": decision_counts,
        "mean_firing_rate": finite_scalar(torch.stack(firing_rates).mean(), "firing_rate"),
        "min_layer_firing_rate": finite_scalar(
            torch.stack(min_layer_firing_rates).amin(), "min_layer_firing_rate"),
        "max_layer_firing_rate": finite_scalar(
            torch.stack(max_layer_firing_rates).amax(), "max_layer_firing_rate"),
        "max_voltage": finite_scalar(torch.stack(voltages).amax(), "max_voltage"),
        "seconds": time.perf_counter() - started,
    }
    return slot, row


def deterministic_batch(split: tuple[Tensor, Tensor], batch_size: int,
                        seed: int) -> tuple[Tensor, Tensor]:
    generator = core.cuda_generator(split[0].device, seed)
    indices = torch.randint(0, len(split[0]), (batch_size,),
                            device=split[0].device, generator=generator)
    return split[0][indices], split[1][indices]


def new_method_state(method: str, snn: core.SNNConfig, device: torch.device,
                     init_seed: int, task: core.SyntheticTask,
                     batch_size: int, method_seed: int) -> dict[str, Any]:
    parameters = core.initialize_snn(snn, device, init_seed)
    first = deterministic_batch(task.train, batch_size, method_seed)
    initial_loss = core.loss_only(parameters, first, snn)
    if method == "stdp":
        optimizer_state = core.OptimizerState.zeros(parameters[-2:], initial_loss)
    else:
        optimizer_state = core.OptimizerState.zeros(parameters, initial_loss)
    return {
        "method": method,
        "step": 0,
        "parameters": core.clone_detached(parameters),
        "optimizer_state": pack_optimizer_state(optimizer_state),
        "loss_sum": 0.0,
        "loss_curve": [],
        "accepted": 0,
        "fallbacks": 0,
        "rollbacks": 0,
        "firing_violations": 0,
        "min_layer_firing_rate": None,
        "max_layer_firing_rate": None,
        "max_voltage": 0.0,
        "min_post_layer_firing_rate": None,
        "max_post_layer_firing_rate": None,
        "max_post_voltage": None,
        "elapsed_seconds": 0.0,
        "peak_incremental_bytes": 0,
    }


def method_step(
    method_state: dict[str, Any], policy: core.AlphaZeroPolicyOptimizer | None,
    task: core.SyntheticTask, snn: core.SNNConfig,
    config: ProductionConfig, inner_lr: float, stdp_lr: float,
    method_seed: int,
) -> tuple[dict[str, Any], dict[str, float | int | str]]:
    started = time.perf_counter()
    method = str(method_state["method"])
    step = int(method_state["step"])
    parameters = tuple(method_state["parameters"])
    state = unpack_optimizer_state(method_state["optimizer_state"])
    batch = deterministic_batch(
        task.train, config.batch_size, method_seed * 1_000_003 + step)
    safety = core.SafetyConfig(max_policy_residual_ratio=config.residual_ratio)
    decision = "accepted"
    telemetry_trace: core.SNNTrace
    post_update_trace: core.SNNTrace | None = None

    if method == "bptt":
        loss, gradients, trace = core.loss_and_gradients(
            parameters, batch, snn, collect_eligibility=False)
        adam = core.adam_proposal(gradients, state, inner_lr)
        updates = tuple(core._clip_update(update, parameter, safety)
                        for update, parameter in zip(adam.updates, parameters))
        parameters = core._project_parameters(
            tuple(parameter + update for parameter, update in zip(parameters, updates)), safety)
        feedback = core.Feedback.initial(loss.detach())
        state = core.OptimizerState(
            tuple(item.detach() for item in adam.momentum),
            tuple(item.detach() for item in adam.variance),
            tuple(item.detach() for item in updates), state.step + 1, feedback)
        telemetry_trace = trace
    elif method == "adam_guarded":
        loss, gradients, trace = core.loss_and_gradients(
            parameters, batch, snn, collect_eligibility=False)
        adam = core.adam_proposal(gradients, state, inner_lr)
        updates = tuple(core._clip_update(update, parameter, safety)
                        for update, parameter in zip(adam.updates, parameters))
        candidate = core._project_parameters(
            tuple(parameter + update
                  for parameter, update in zip(parameters, updates)), safety)
        committed_updates = tuple(
            new - old for new, old in zip(candidate, parameters))
        next_state = core.OptimizerState(
            tuple(item.detach() for item in adam.momentum),
            tuple(item.detach() for item in adam.variance),
            tuple(item.detach() for item in committed_updates),
            state.step + 1, state.feedback)
        zero = torch.zeros((), device=parameters[0].device)
        proposal = core.LearnedProposal(
            candidate, committed_updates, candidate, committed_updates,
            state, next_state, zero, zero)
        guard_batch = deterministic_batch(
            task.train, config.batch_size,
            method_seed * 2_000_003 + 500_009 + step)
        outcome = core.guarded_commit(
            parameters, proposal, guard_batch, snn, safety,
            progress=(step + 1) / config.deployment_steps)
        parameters, state = outcome.parameters, outcome.state
        telemetry_trace = trace
        post_update_trace = outcome.trace
        decision = ("accepted" if outcome.accepted_policy else
                    "adam_fallback" if outcome.used_adam_fallback else "rollback")
    elif method == "stdp":
        loss, readout_gradients, trace = core.loss_and_readout_gradients(
            parameters, batch, snn)
        adam = core.adam_proposal(readout_gradients, state, inner_lr)
        updates: list[Tensor] = []
        final_weight = 2 * (snn.trainable_layers - 1)
        for index, parameter in enumerate(parameters):
            if index % 2 == 0 and index < final_weight:
                update = stdp_lr * core._matched_basis(
                    trace.eligibility[index // 2], parameter)
            elif index < final_weight:
                update = torch.zeros_like(parameter)
            else:
                update = adam.updates[index - final_weight]
            updates.append(core._clip_update(update, parameter, safety))
        parameters = core._project_parameters(
            tuple(parameter + update for parameter, update in zip(parameters, updates)), safety)
        state = core.OptimizerState(
            tuple(item.detach() for item in adam.momentum),
            tuple(item.detach() for item in adam.variance),
            tuple(item.detach() for item in updates[-2:]), state.step + 1,
            core.Feedback.initial(loss.detach()))
        telemetry_trace = trace
    elif method in {"zero_shot_unguarded", "policy_unguarded", "policy_guarded"}:
        if policy is None:
            raise ValueError(f"{method} requires a policy network")
        loss, gradients, trace = core.loss_and_gradients(parameters, batch, snn)
        policy.eval()
        with torch.no_grad():
            proposal = core.propose_learned_update(
                policy, parameters, gradients, trace, state, inner_lr,
                safety, amp=True, residual_scale=1.0)
        guard_batch = deterministic_batch(
            task.train, config.batch_size,
            method_seed * 2_000_003 + 500_009 + step)
        if method == "policy_guarded":
            outcome = core.guarded_commit(
                parameters, proposal, guard_batch, snn, safety,
                progress=(step + 1) / config.deployment_steps)
            parameters, state = outcome.parameters, outcome.state
            telemetry_trace = trace
            post_update_trace = outcome.trace
            decision = ("accepted" if outcome.accepted_policy else
                        "adam_fallback" if outcome.used_adam_fallback else "rollback")
        else:
            with torch.no_grad():
                before, _ = core._guard_metrics(parameters, guard_batch, snn)
                after, after_trace = core._guard_metrics(
                    proposal.parameters, guard_batch, snn)
                if not bool(torch.isfinite(after)):
                    raise FloatingPointError("unguarded policy produced non-finite loss")
                reward = (before.clamp_min(1e-8).log()
                          - after.clamp_min(1e-8).log()).clamp(-1, 1)
                feedback = core.Feedback(
                    after.detach(),
                    (0.95 * state.feedback.loss_ema + 0.05 * after).detach(),
                    reward.detach(), (state.feedback.loss - after).detach(),
                    torch.ones_like(after), torch.zeros_like(after),
                    torch.zeros_like(after),
                    core.scalar((step + 1) / config.deployment_steps, after))
            parameters = core.clone_detached(proposal.parameters)
            state = dataclasses.replace(proposal.next_state, feedback=feedback).detached()
            telemetry_trace = trace
            post_update_trace = after_trace
    else:
        raise ValueError(f"unknown evaluation method {method!r}")

    firing_rate = finite_scalar(
        telemetry_trace.mean_spike_rate, f"{method}_firing_rate")
    min_layer_rate, max_layer_rate = hidden_layer_rate_stats(telemetry_trace)
    min_layer_firing_rate = finite_scalar(
        min_layer_rate, f"{method}_min_layer_firing_rate")
    max_layer_firing_rate = finite_scalar(
        max_layer_rate, f"{method}_max_layer_firing_rate")
    max_voltage = finite_scalar(
        telemetry_trace.max_abs_voltage, f"{method}_max_voltage")
    post_update_observed = post_update_trace is not None
    safety_trace = post_update_trace if post_update_trace is not None else telemetry_trace
    safety_min, safety_max = hidden_layer_rate_stats(safety_trace)
    post_min_layer_firing_rate = finite_scalar(
        safety_min, f"{method}_post_min_layer_firing_rate")
    post_max_layer_firing_rate = finite_scalar(
        safety_max, f"{method}_post_max_layer_firing_rate")
    post_max_voltage = finite_scalar(
        safety_trace.max_abs_voltage, f"{method}_post_max_voltage")
    firing_alarm = bool(
        post_min_layer_firing_rate < safety.min_spike_rate
        or post_max_layer_firing_rate > safety.max_spike_rate)
    loss_value = finite_scalar(loss, f"{method}_loss")
    method_state = dict(method_state)
    method_state["parameters"] = core.clone_detached(parameters)
    method_state["optimizer_state"] = pack_optimizer_state(state.detached())
    method_state["step"] = step + 1
    method_state["loss_sum"] = float(method_state["loss_sum"]) + loss_value
    method_state["loss_curve"] = [*method_state["loss_curve"], loss_value]
    method_state["accepted"] = int(method_state["accepted"]) + int(decision == "accepted")
    method_state["fallbacks"] = int(method_state["fallbacks"]) + int(decision == "adam_fallback")
    method_state["rollbacks"] = int(method_state["rollbacks"]) + int(decision == "rollback")
    method_state["firing_violations"] = (
        int(method_state.get("firing_violations", 0)) + int(firing_alarm))
    previous_min = method_state.get("min_layer_firing_rate")
    previous_max = method_state.get("max_layer_firing_rate")
    method_state["min_layer_firing_rate"] = (
        min_layer_firing_rate if previous_min is None
        else min(float(previous_min), min_layer_firing_rate))
    method_state["max_layer_firing_rate"] = (
        max_layer_firing_rate if previous_max is None
        else max(float(previous_max), max_layer_firing_rate))
    method_state["max_voltage"] = max(
        float(method_state.get("max_voltage", 0.0)), max_voltage)
    if post_update_observed:
        previous_post_min = method_state.get("min_post_layer_firing_rate")
        previous_post_max = method_state.get("max_post_layer_firing_rate")
        previous_post_voltage = method_state.get("max_post_voltage")
        method_state["min_post_layer_firing_rate"] = (
            post_min_layer_firing_rate if previous_post_min is None
            else min(float(previous_post_min), post_min_layer_firing_rate))
        method_state["max_post_layer_firing_rate"] = (
            post_max_layer_firing_rate if previous_post_max is None
            else max(float(previous_post_max), post_max_layer_firing_rate))
        method_state["max_post_voltage"] = (
            post_max_voltage if previous_post_voltage is None
            else max(float(previous_post_voltage), post_max_voltage))
    duration = time.perf_counter() - started
    method_state["elapsed_seconds"] = float(method_state["elapsed_seconds"]) + duration
    row: dict[str, float | int | str] = {
        "method": method, "step": step + 1, "loss": loss_value,
        "decision": decision, "firing_rate": firing_rate,
        "min_layer_firing_rate": min_layer_firing_rate,
        "max_layer_firing_rate": max_layer_firing_rate,
        "post_min_layer_firing_rate": (
            post_min_layer_firing_rate if post_update_observed else None),
        "post_max_layer_firing_rate": (
            post_max_layer_firing_rate if post_update_observed else None),
        "firing_alarm": firing_alarm,
        "max_voltage": max_voltage,
        "post_max_voltage": post_max_voltage if post_update_observed else None,
        "seconds": duration,
    }
    return method_state, row


def finalize_method(method_state: dict[str, Any], task: core.SyntheticTask,
                    snn: core.SNNConfig) -> dict[str, Any]:
    parameters = tuple(method_state["parameters"])
    validation = core.evaluate_snn(parameters, task.validation, snn)
    test = core.evaluate_snn(parameters, task.test, snn)
    steps = max(1, int(method_state["step"]))
    return {
        "method": method_state["method"],
        "validation": validation,
        "test": test,
        "train_loss_auc": float(method_state["loss_sum"]) / steps,
        "loss_curve": list(method_state["loss_curve"]),
        "accepted": int(method_state["accepted"]),
        "fallbacks": int(method_state["fallbacks"]),
        "rollbacks": int(method_state["rollbacks"]),
        "firing_violations": int(method_state.get("firing_violations", 0)),
        "min_layer_firing_rate": method_state.get("min_layer_firing_rate"),
        "max_layer_firing_rate": method_state.get("max_layer_firing_rate"),
        "max_voltage": float(method_state.get("max_voltage", 0.0)),
        "min_post_layer_firing_rate": method_state.get(
            "min_post_layer_firing_rate"),
        "max_post_layer_firing_rate": method_state.get(
            "max_post_layer_firing_rate"),
        "max_post_voltage": method_state.get("max_post_voltage"),
        "elapsed_seconds": float(method_state["elapsed_seconds"]),
    }


def aggregate_results(
    rows: Sequence[dict[str, Any]],
    config: ProductionConfig,
    capacity_bounds: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["axis"]), str(row["spec"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    capacity = dict(capacity_bounds or {})
    output: dict[str, Any] = {
        "rows": [],
        "paired_effects": [],
        "bounds": {},
        "capacity_bounds": capacity,
        "bound_thresholds": {
            "loss_gap": config.stop_gap,
            "guard_rejection_rate": config.stop_rejection_rate,
            "consecutive_sizes": config.stop_patience,
            "learnability_improvement": config.learnability_improvement,
        },
    }
    for (axis, spec, method), items in sorted(grouped.items()):
        summary = {
            "axis": axis, "spec": spec, "method": method,
            "snn_parameters": items[0]["snn_parameters"],
            "validation_loss": mean_ci95([item["validation"]["loss"] for item in items]),
            "validation_accuracy": mean_ci95(
                [item["validation"]["accuracy"] for item in items]),
            "test_loss": mean_ci95([item["test"]["loss"] for item in items]),
            "test_accuracy": mean_ci95([item["test"]["accuracy"] for item in items]),
            "train_loss_auc": mean_ci95([item["train_loss_auc"] for item in items]),
            "elapsed_seconds": mean_ci95([item["elapsed_seconds"] for item in items]),
            "rejection_rate": mean_ci95([
                (item["fallbacks"] + item["rollbacks"])
                / max(1, item["accepted"] + item["fallbacks"] + item["rollbacks"])
                for item in items]),
            "firing_violation_rate": mean_ci95([
                item.get("firing_violations", 0)
                / max(1, len(item.get("loss_curve", ())))
                for item in items]),
            "min_layer_firing_rate": mean_ci95([
                item.get("min_layer_firing_rate") for item in items]),
            "max_layer_firing_rate": mean_ci95([
                item.get("max_layer_firing_rate") for item in items]),
            "max_voltage": mean_ci95([
                item.get("max_voltage") for item in items]),
            "min_post_layer_firing_rate": mean_ci95([
                item.get("min_post_layer_firing_rate") for item in items]),
            "max_post_layer_firing_rate": mean_ci95([
                item.get("max_post_layer_firing_rate") for item in items]),
            "max_post_voltage": mean_ci95([
                item.get("max_post_voltage") for item in items]),
        }
        output["rows"].append(summary)

    comparisons = (
        ("policy_unguarded", "zero_shot_unguarded"),
        ("policy_unguarded", "bptt"),
        ("policy_unguarded", "stdp"),
        ("policy_guarded", "policy_unguarded"),
    )
    for axis, spec in sorted({(str(row["axis"]), str(row["spec"]))
                              for row in rows}):
        by_method = {
            method: [row for row in rows
                     if row["axis"] == axis and row["spec"] == spec
                     and row["method"] == method]
            for pair in comparisons for method in pair
        }
        for left_method, right_method in comparisons:
            left, right = by_method[left_method], by_method[right_method]
            if not left or not right:
                continue
            output["paired_effects"].append({
                "axis": axis,
                "spec": spec,
                "left_method": left_method,
                "right_method": right_method,
                "validation_loss": paired_metric_summary(
                    left, right, "validation", "loss"),
                "validation_accuracy": paired_metric_summary(
                    left, right, "validation", "accuracy"),
                "test_loss": paired_metric_summary(
                    left, right, "test", "loss"),
                "test_accuracy": paired_metric_summary(
                    left, right, "test", "accuracy"),
            })

    # Primary effect: frozen unguarded policy CE relative to projected-Adam/BPTT.
    for axis in {key[0] for key in grouped}:
        bound_rows = []
        specs = []
        for row in rows:
            if row["axis"] == axis and row["spec"] not in specs:
                specs.append(row["spec"])
        consecutive = 0
        found = None
        for spec in specs:
            by_method = {
                method: [item for item in rows
                         if item["axis"] == axis and item["spec"] == spec
                         and item["method"] == method]
                for method in ("untrained", "bptt", "zero_shot_unguarded",
                               "policy_unguarded", "policy_guarded")
            }
            if any(not values for values in by_method.values()):
                continue
            untrained = statistics.fmean(item["validation"]["loss"]
                                         for item in by_method["untrained"])
            bptt = statistics.fmean(item["validation"]["loss"]
                                    for item in by_method["bptt"])
            zero_shot = statistics.fmean(
                item["validation"]["loss"]
                for item in by_method["zero_shot_unguarded"])
            policy = statistics.fmean(item["validation"]["loss"]
                                      for item in by_method["policy_unguarded"])
            guarded_reject = statistics.fmean(
                (item["fallbacks"] + item["rollbacks"])
                / max(1, item["accepted"] + item["fallbacks"] + item["rollbacks"])
                for item in by_method["policy_guarded"])
            learnable = bptt < (1.0 - config.learnability_improvement) * untrained
            gap = ((policy - zero_shot) / max(1e-8, untrained - bptt)
                   if learnable else None)
            failed = bool(
                learnable and gap is not None and gap > config.stop_gap
            ) or guarded_reject > config.stop_rejection_rate
            consecutive = consecutive + 1 if failed else 0
            bound_rows.append({"spec": spec, "learnable": learnable,
                               "loss_gap": gap,
                               "zero_shot_bptt_loss_delta": zero_shot - bptt,
                               "guard_rejection_rate": guarded_reject})
            if consecutive >= config.stop_patience and found is None:
                found = spec
        output["bounds"][axis] = {
            "reached": found is not None, "failing_size": found,
            "rows": bound_rows,
        }
    # Hardware capacity is distinct from a learned-optimizer quality bound.  It
    # is nevertheless part of the aggregate so a run that exhausts VRAM before
    # producing any rows for an axis still has a complete, explicit result.
    for axis, record in capacity.items():
        axis_bound = output["bounds"].setdefault(
            axis, {"reached": False, "failing_size": None, "rows": []})
        axis_bound["capacity_bound"] = record
    return output


class RunContext:
    def __init__(
        self, run_dir: Path, device: torch.device, config: ProductionConfig,
        core_manifest: dict[str, Any], production_manifest: dict[str, Any], *, resume: bool,
        checkpoint_every: int, heartbeat_seconds: float,
        log_every: int,
    ) -> None:
        self.run_dir = run_dir
        self.device = device
        self.config = config
        self.source_sha256 = production_source_digest()
        self.core_manifest = core_manifest
        self.production_manifest = production_manifest
        self.production_manifest_sha256 = stable_config_digest(production_manifest)
        self.checkpoint_every = checkpoint_every
        self.heartbeat_seconds = heartbeat_seconds
        self.log_every = log_every
        self.last_checkpoint_step = -1
        self.last_checkpoint_time = 0.0
        self.last_heartbeat = 0.0
        self.policy: core.AlphaZeroPolicyOptimizer | None = None
        self.zero_shot_policy: core.AlphaZeroPolicyOptimizer | None = None
        self.outer_optimizer: torch.optim.Optimizer | None = None
        self.state: dict[str, Any]
        self.run_id: str

        self.checkpoint_config = {
            "schema": PRODUCTION_SCHEMA,
            "source_sha256": self.source_sha256,
            "core_manifest_sha256": stable_config_digest(core_manifest),
            "production_manifest_sha256": self.production_manifest_sha256,
            "device": core.verification_identity(device),
            "scientific_config": dataclasses.asdict(config),
        }
        self.config_sha256 = stable_config_digest(self.checkpoint_config)
        run_dir.mkdir(parents=True, exist_ok=True)

        if resume:
            bootstrap_manager = CheckpointManager(
                run_dir / "checkpoints", keep_last=3)
            loaded = bootstrap_manager.load_latest(
                map_location=device, expected_config=self.checkpoint_config)
            payload = loaded.payload
            self.run_id = str(payload["run_id"])
            self.state = payload["state"]
            reconcile_terminal_status(
                self.state, self.run_id, run_dir / "status.json")
            if self.state.get("status") not in ("complete", "error"):
                self.state["status"] = "running"
        else:
            existing = [path for path in run_dir.iterdir()
                        if path.name != "run.lock"]
            if existing:
                raise RuntimeError(
                    f"run directory is not empty: {run_dir}; use --resume or a new path")
            self.run_id = str(uuid.uuid4())
            self.state = initial_run_state(self.run_id, config)

        archived_manifest = run_dir / "production_verification.json"
        if resume:
            if not archived_manifest.is_file():
                raise RuntimeError("run-local production verification manifest is missing")
            archived = json.loads(archived_manifest.read_text())
            if stable_config_digest(archived) != self.production_manifest_sha256:
                raise RuntimeError("run-local production verification manifest changed")

        self.logger = EventLogger(
            run_dir / "events.jsonl", self.run_id,
            context={"source_sha256": self.source_sha256,
                     "config_sha256": self.config_sha256,
                     "device": str(device)})
        self.checkpoints = CheckpointManager(
            run_dir / "checkpoints", keep_last=3, logger=self.logger)
        self.best_checkpoints = CheckpointManager(
            run_dir / "best", prefix="best-policy", keep_last=3, logger=self.logger)
        self.zero_shot_checkpoints = CheckpointManager(
            run_dir / "zero_shot", prefix="zero-shot-policy", keep_last=2,
            logger=self.logger)
        self.frozen_checkpoints = CheckpointManager(
            run_dir / "frozen", prefix="frozen-policy", keep_last=2,
            logger=self.logger)
        self.signal = SignalController(
            on_graceful=lambda signum: self.logger.log(
                "signal_stop_requested", level="warning", critical=True,
                signal=signum),
            on_checkpoint=lambda signum: self.logger.log(
                "signal_checkpoint_requested", level="warning", critical=True,
                signal=signum),
            on_forced=lambda signum: self.logger.log(
                "signal_force_requested", level="error", critical=True,
                signal=signum),
        )

        if resume:
            # The manager loaded tensors directly onto the selected device.
            payload = self.checkpoints.load_latest(
                map_location=device, expected_config=self.checkpoint_config).payload
            if (payload.get("policy") is not None
                    or self.state.get("frozen_policy_milestone") is not None):
                self._restore_model(payload)
            torch.set_rng_state(payload["cpu_rng"].cpu())
            torch.cuda.set_rng_state_all(
                [value.cpu() for value in payload["cuda_rng"]])
            self.logger.log("run_resumed", critical=True,
                            phase=self.state.get("phase"),
                            meta_step=self.state.get("meta_step"))
        else:
            atomic_write_json(run_dir / "config.json", self.checkpoint_config)
            atomic_write_json(archived_manifest, production_manifest)
            self.logger.log("run_created", critical=True,
                            run_id=self.run_id,
                            config=self.checkpoint_config,
                            gpu=gpu_telemetry(device))

    def _new_policy(self) -> core.AlphaZeroPolicyOptimizer:
        with torch.device(self.device):
            return core.AlphaZeroPolicyOptimizer()

    def ensure_policy(self) -> None:
        if self.policy is not None:
            return
        core.seed_everything(self.config.policy_seed)
        self.policy = self._new_policy()
        self.outer_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.config.meta_lr,
            fused=True)

    def preserve_zero_shot(self) -> None:
        """Durably preserve the exact meta-step-zero policy once per run."""
        assert self.policy is not None
        if self.state.get("zero_shot_policy_milestone") is not None:
            return
        if int(self.state.get("meta_step", 0)) != 0:
            raise HealthAbort("zero-shot policy was not preserved before meta-training")
        payload = {"policy": self.policy.state_dict(), "meta_step": 0}
        for replica in (1, 2):
            record = self.zero_shot_checkpoints.save(
                payload, config=self.checkpoint_config,
                extra_metadata={"kind": "zero_shot", "meta_step": 0,
                                "replica": replica})
        self.state["zero_shot_policy_milestone"] = record.generation
        self.event("zero_shot_policy_preserved", critical=True,
                   generation=record.generation)

    def ensure_zero_shot_policy(self) -> None:
        if self.zero_shot_policy is not None:
            return
        if self.state.get("zero_shot_policy_milestone") is None:
            raise HealthAbort("zero-shot policy snapshot is missing")
        loaded = self.zero_shot_checkpoints.load_latest(
            map_location=self.device, expected_config=self.checkpoint_config)
        self.zero_shot_policy = self._new_policy()
        self.zero_shot_policy.load_state_dict(loaded.payload["policy"])
        self.zero_shot_policy.eval()

    def _restore_model(self, payload: dict[str, Any]) -> None:
        if payload.get("policy") is not None:
            self.policy = self._new_policy()
            self.policy.load_state_dict(payload["policy"])
            self.outer_optimizer = torch.optim.AdamW(
                self.policy.parameters(), lr=self.config.meta_lr,
                fused=True)
            if payload.get("outer_optimizer") is not None:
                self.outer_optimizer.load_state_dict(payload["outer_optimizer"])
            return
        frozen = self.frozen_checkpoints.load_latest(
            map_location=self.device, expected_config=self.checkpoint_config)
        self.policy = self._new_policy()
        self.policy.load_state_dict(frozen.payload["policy"])
        self.outer_optimizer = None

    def payload(self, *, include_policy: bool) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_SCHEMA,
            "run_id": self.run_id,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "state": self.state,
            "policy": (self.policy.state_dict()
                       if include_policy and self.policy is not None else None),
            "outer_optimizer": (
                self.outer_optimizer.state_dict()
                if include_policy and self.outer_optimizer is not None else None),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }

    def checkpoint(self, reason: str, *, include_policy: bool = True,
                   force: bool = False) -> None:
        if production_source_digest() != self.source_sha256:
            raise HealthAbort("production source changed while the run was active")
        meta_step = int(self.state.get("meta_step", 0))
        now = time.monotonic()
        if (not force and meta_step - self.last_checkpoint_step < self.checkpoint_every
                and now - self.last_checkpoint_time < 300):
            return
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        generation = self.checkpoints.next_generation()
        previous_generation = self.state.get("last_checkpoint_generation")
        self.state["last_checkpoint_generation"] = generation
        try:
            record = self.checkpoints.save(
                self.payload(include_policy=include_policy),
                generation=generation,
                config=self.checkpoint_config,
                extra_metadata={"reason": reason, "phase": self.state.get("phase"),
                                "meta_step": meta_step})
        except BaseException:
            self.state["last_checkpoint_generation"] = previous_generation
            raise
        self.last_checkpoint_step = meta_step
        self.last_checkpoint_time = now
        self.event("checkpoint_complete", critical=True, reason=reason,
                   generation=record.generation,
                   seconds=time.perf_counter() - started,
                   size_bytes=record.metadata["size_bytes"])

    def save_best(self, dev_loss: float) -> None:
        assert self.policy is not None
        selected_meta_step = int(self.state["meta_step"])
        payload = {
            "policy": self.policy.state_dict(),
            "outer_optimizer": (self.outer_optimizer.state_dict()
                                if self.outer_optimizer is not None else None),
            "meta_step": selected_meta_step, "dev_loss": dev_loss,
        }
        for replica in (1, 2):
            record = self.best_checkpoints.save(
                payload, config=self.checkpoint_config,
                extra_metadata={"kind": "best", "dev_loss": dev_loss,
                                "meta_step": selected_meta_step,
                                "replica": replica})
        self.state["best_policy_milestone"] = record.generation
        self.state["best_meta_step"] = selected_meta_step

    def freeze_best(self) -> None:
        selected_meta_step = int(self.state["meta_step"])
        selected_dev_loss = self.state.get("best_dev_loss")
        if self.state.get("best_policy_milestone") is not None:
            loaded = self.best_checkpoints.load_latest(
                map_location=self.device, expected_config=self.checkpoint_config)
            assert self.policy is not None
            self.policy.load_state_dict(loaded.payload["policy"])
            selected_meta_step = int(loaded.payload["meta_step"])
            selected_dev_loss = float(loaded.payload["dev_loss"])
            self.state["best_meta_step"] = selected_meta_step
            self.state["best_dev_loss"] = selected_dev_loss
        assert self.policy is not None
        payload = {
            "policy": self.policy.state_dict(),
            "selected_meta_step": selected_meta_step,
            "best_dev_loss": selected_dev_loss,
        }
        for replica in (1, 2):
            record = self.frozen_checkpoints.save(
                payload, config=self.checkpoint_config,
                extra_metadata={"kind": "frozen",
                                "selected_meta_step": selected_meta_step,
                                "best_dev_loss": selected_dev_loss,
                                "replica": replica})
        self.state["frozen_policy_milestone"] = record.generation
        self.outer_optimizer = None

    def event(self, event: str, *, level: str = "info",
              critical: bool = False, **data: Any) -> None:
        envelope = {
            "run_phase": self.state.get("phase"),
            "run_meta_step": self.state.get("meta_step"),
            "gpu": gpu_telemetry(self.device),
            **data,
        }
        self.logger.log(event, level=level, critical=critical, **envelope)
        self._write_status(event, level, data)

    def _write_status(self, event: str, level: str, data: dict[str, Any]) -> None:
        atomic_write_json(self.run_dir / "status.json", {
            "schema": PRODUCTION_SCHEMA,
            "run_id": self.run_id,
            "timestamp_utc": utc_timestamp(),
            "status": self.state.get("status"),
            "phase": self.state.get("phase"),
            "meta_step": self.state.get("meta_step"),
            "target_meta_steps": self.config.meta_max_steps,
            "training_termination": self.state.get("training_termination"),
            "early_stop": self.state.get("early_stop"),
            "alarm_summary": alarm_summary(self.state.get("alarms", ())),
            "last_event": event,
            "level": level,
            "data": data,
            "gpu": gpu_telemetry(self.device),
        })

    def safe_point(self, *, include_policy: bool = True,
                   force_checkpoint: bool = False) -> None:
        telemetry = gpu_telemetry(self.device)
        total = int(telemetry.get("total_bytes", 0))
        fractions: dict[str, float] = {}
        if total > 0:
            free = int(telemetry.get("free_bytes", total))
            fractions = {
                "allocated": int(telemetry.get("allocated_bytes", 0)) / total,
                "peak_allocated": int(
                    telemetry.get("max_allocated_bytes", 0)) / total,
                "reserved": int(telemetry.get("reserved_bytes", 0)) / total,
                "peak_reserved": int(
                    telemetry.get("max_reserved_bytes", 0)) / total,
                "device_used": max(0, total - free) / total,
            }
        exceeded = {name: value for name, value in fractions.items()
                    if value > self.config.max_vram_fraction}
        if exceeded:
            alarm = {"type": "vram",
                     "threshold": self.config.max_vram_fraction,
                     "exceeded": exceeded, "fractions": fractions,
                     "telemetry": telemetry}
            self.state["alarms"].append(alarm)
            self.event("vram_limit_exceeded", level="error", critical=True,
                       threshold=self.config.max_vram_fraction,
                       exceeded=exceeded, fractions=fractions,
                       telemetry_at_alarm=telemetry)
            self.checkpoint("vram_alarm", include_policy=include_policy, force=True)
            raise HealthAbort(
                "VRAM crossed the production limit: "
                + ", ".join(f"{name}={value:.3f}"
                            for name, value in exceeded.items()))
        now = time.monotonic()
        if now - self.last_heartbeat >= self.heartbeat_seconds:
            current = self.state.get("current_method")
            progress = None
            if current is not None:
                progress = {
                    "identity": current.get("identity"),
                    "method": current.get("method"),
                    "step": current.get("step"),
                    "loss_sum": current.get("loss_sum"),
                    "accepted": current.get("accepted"),
                    "fallbacks": current.get("fallbacks"),
                    "rollbacks": current.get("rollbacks"),
                }
            self.event("heartbeat", progress=progress,
                       cursor=self.state.get("evaluation_cursor"))
            self.last_heartbeat = now
        requested = self.signal.consume_checkpoint_request()
        if self.signal.stop_requested.is_set():
            self.state["status"] = "stopped"
            self.checkpoint("graceful_stop", include_policy=include_policy, force=True)
            self.event("graceful_stop", level="warning", critical=True,
                       signal=self.signal.last_signal)
            raise GracefulStop("graceful stop requested")
        if requested or force_checkpoint:
            self.checkpoint("signal" if requested else "periodic",
                            include_policy=include_policy, force=True)

    def close(self) -> None:
        self.logger.close()


def run_task_controls(ctx: RunContext) -> None:
    ctx.state["phase"] = "task_controls"
    controls = ctx.state.setdefault("task_controls", {})
    seeds = (*ctx.config.dev_task_seeds, *ctx.config.eval_task_seeds)
    for seed in seeds:
        key = str(seed)
        if key in controls:
            continue
        ctx.safe_point(include_policy=False)
        task = core.make_synthetic_task(ctx.config.task_config(), ctx.device, seed)
        result = core.task_suitability_check(task, ctx.config.task_config(), steps=100)
        controls[key] = result
        ctx.event("task_control_complete", seed=seed, controls=result)
        ctx.checkpoint("task_control", include_policy=False, force=True)
    ctx.state["phase"] = "tuning"
    ctx.checkpoint("task_controls_complete", include_policy=False, force=True)


def _tuning_epochs(config: ProductionConfig) -> int:
    steps_per_epoch = math.ceil(config.train_samples / config.batch_size)
    return max(1, math.ceil(config.tune_steps / steps_per_epoch))


def tuning_specs(config: ProductionConfig) -> tuple[str, ...]:
    """Architectures whose deployment learning rates are locked on dev seeds."""
    return tuple(dict.fromkeys(
        (*config.meta_train_specs, *config.width_specs, *config.depth_specs,
         "2x128")))


def selected_learning_rate(state: Mapping[str, Any], method: str,
                           spec: str) -> float:
    selected = state["selected_lrs"][method]
    if not isinstance(selected, Mapping) or spec not in selected:
        raise HealthAbort(f"missing tuned {method} learning rate for {spec}")
    return float(selected[spec])


def run_lr_tuning(ctx: RunContext) -> None:
    ctx.state["phase"] = "tuning"
    tuning = ctx.state.setdefault("tuning", {})
    task_config = ctx.config.task_config()
    epochs = _tuning_epochs(ctx.config)
    safety = core.SafetyConfig(max_policy_residual_ratio=ctx.config.residual_ratio)

    for method, learning_rates in (("bptt", ctx.config.bptt_lrs),
                                   ("policy_base", ctx.config.bptt_lrs),
                                   ("stdp", ctx.config.stdp_lrs)):
        selected_by_spec = ctx.state["selected_lrs"].setdefault(method, {})
        if not isinstance(selected_by_spec, dict):
            raise HealthAbort("selected learning-rate state has an invalid schema")
        for spec_index, spec in enumerate(tuning_specs(ctx.config)):
            snn = ctx.config.snn(spec)
            for learning_rate in learning_rates:
                for repeat, seed in enumerate(ctx.config.dev_task_seeds):
                    key = f"{method}:{spec}:{learning_rate:.9g}:{seed}"
                    if key in tuning:
                        continue
                    ctx.safe_point(include_policy=False)
                    ctx.event("tuning_candidate_start", method=method, spec=spec,
                              learning_rate=learning_rate, seed=seed)
                    task = core.make_synthetic_task(task_config, ctx.device, seed)
                    init_seed = (ctx.config.policy_seed * 100_003
                                 + spec_index * 1009 + repeat)
                    initial = core.initialize_snn(snn, ctx.device, init_seed)
                    rollbacks = 0
                    fallbacks = 0
                    if method == "policy_base":
                        guarded = new_method_state(
                            "adam_guarded", snn, ctx.device, init_seed, task,
                            ctx.config.batch_size, seed + 17)
                        for _ in range(ctx.config.tune_steps):
                            guarded, _ = method_step(
                                guarded, None, task, snn, ctx.config,
                                learning_rate, 0.0, seed + 17)
                        result = finalize_method(guarded, task, snn)
                        metrics = result["validation"]
                        seconds = float(result["elapsed_seconds"])
                        rollbacks = int(result["rollbacks"])
                        fallbacks = int(result["fallbacks"])
                    else:
                        train = core.TrainConfig(
                            batch_size=ctx.config.batch_size, epochs=epochs,
                            inner_lr=(
                                learning_rate if method == "bptt" else
                                selected_learning_rate(
                                    ctx.state, "bptt", spec)),
                            stdp_lr=learning_rate, meta_steps=1, unroll=1,
                            episode_length=1, amp=True)
                    if method == "bptt":
                        trained = core.train_surrogate_bptt(
                            initial, task.train, snn, train, safety, seed=seed + 17)
                    elif method == "stdp":
                        trained = core.train_stdp(
                            initial, task.train, snn, train, safety, seed=seed + 17)
                    if method != "policy_base":
                        metrics = core.evaluate_snn(
                            trained.parameters, task.validation, snn)
                        seconds = trained.elapsed_seconds
                    tuning[key] = {
                        "method": method, "spec": spec,
                        "learning_rate": learning_rate, "seed": seed,
                        "validation": metrics,
                        "seconds": seconds, "rollbacks": rollbacks,
                        "fallbacks": fallbacks,
                    }
                    ctx.event("tuning_candidate_complete", method=method, spec=spec,
                              learning_rate=learning_rate, seed=seed,
                              metrics=metrics, seconds=seconds,
                              rollbacks=rollbacks, fallbacks=fallbacks)
                    ctx.checkpoint(
                        "tuning_candidate", include_policy=False, force=True)

            candidates: dict[float, list[float]] = {}
            for row in tuning.values():
                if row["method"] == method and row.get("spec") == spec:
                    candidates.setdefault(float(row["learning_rate"]), []).append(
                        float(row["validation"]["loss"]))
            rollback_rates: dict[float, float] = {}
            for learning_rate in candidates:
                matching = [
                    row for row in tuning.values()
                    if row["method"] == method and row.get("spec") == spec
                    and float(row["learning_rate"]) == learning_rate]
                rollback_rates[learning_rate] = (
                    sum(int(row.get("rollbacks", 0)) for row in matching)
                    / max(1, ctx.config.tune_steps * len(matching)))
            eligible = list(candidates)
            if method == "policy_base":
                eligible = [
                    rate for rate in candidates
                    if rollback_rates[rate] <= ctx.config.meta_max_rollback_fraction]
                if not eligible:
                    minimum = min(rollback_rates.values())
                    eligible = [rate for rate in candidates
                                if rollback_rates[rate] == minimum]
                    ctx.event(
                        "policy_base_tuning_no_stable_candidate",
                        level="warning", critical=True, spec=spec,
                        rollback_rates=rollback_rates)
            selected = min(
                eligible,
                key=lambda value: statistics.fmean(candidates[value]))
            selected_by_spec[spec] = selected
            ctx.event("learning_rate_selected", critical=True, method=method,
                      spec=spec, learning_rate=selected,
                      mean_validation_loss=statistics.fmean(candidates[selected]),
                      rollback_rate=rollback_rates[selected])

    ctx.state["phase"] = "meta_training"
    ctx.checkpoint("tuning_complete", include_policy=False, force=True)


def evaluate_policy_on_dev_tasks(
    policy: core.AlphaZeroPolicyOptimizer,
    config: ProductionConfig,
    device: torch.device,
    inner_lr: float,
    stdp_lr: float,
) -> dict[str, Any]:
    """Evaluate one policy on every declared dev task with paired fixed seeds."""
    snn = config.snn("2x128")
    per_seed: list[dict[str, Any]] = []
    for repeat, task_seed in enumerate(config.dev_task_seeds):
        task = core.make_synthetic_task(config.task_config(), device, task_seed)
        # Both seeds are stable across meta checkpoints, distinct across tasks,
        # and independent of the task generator.  This makes checkpoint selection
        # a paired comparison instead of repeatedly sampling evaluation noise.
        init_seed = config.policy_seed * 1_000_003 + 800_011 + repeat * 1009
        method_seed = task_seed * 65_537 + 900_001 + repeat * 101
        state = new_method_state(
            "policy_unguarded", snn, device, init_seed, task,
            config.batch_size, method_seed)
        for _ in range(config.dev_steps):
            state, _ = method_step(
                state, policy, task, snn, config, inner_lr, stdp_lr,
                method_seed)
        result = finalize_method(state, task, snn)
        per_seed.append({
            "task_seed": task_seed,
            "init_seed": init_seed,
            "method_seed": method_seed,
            "validation_loss": result["validation"]["loss"],
            "validation_accuracy": result["validation"]["accuracy"],
            "train_loss_auc": result["train_loss_auc"],
        })
    return {
        "dev_task_count": len(per_seed),
        "validation_loss": statistics.fmean(
            float(row["validation_loss"]) for row in per_seed),
        "validation_accuracy": statistics.fmean(
            float(row["validation_accuracy"]) for row in per_seed),
        "train_loss_auc": statistics.fmean(
            float(row["train_loss_auc"]) for row in per_seed),
        "per_seed": per_seed,
    }


def run_dev_evaluation(ctx: RunContext) -> dict[str, Any]:
    assert ctx.policy is not None
    return evaluate_policy_on_dev_tasks(
        ctx.policy, ctx.config, ctx.device,
        selected_learning_rate(ctx.state, "policy_base", "2x128"),
        selected_learning_rate(ctx.state, "stdp", "2x128"),
    )


def meta_rollback_health(
    history: Sequence[dict[str, Any]], config: ProductionConfig,
) -> dict[str, Any]:
    """Summarize whether safe no-op rollbacks are becoming pathological."""
    recent = list(history[-config.meta_rollback_window:])
    decisions = sum(int(row["horizon"]) for row in recent)
    rollbacks = sum(int(row["rollback_count"]) for row in recent)
    fraction = rollbacks / max(1, decisions)
    consecutive_steps = 0
    for row in reversed(history):
        if int(row["rollback_count"]) == 0:
            break
        consecutive_steps += 1
    warmed_up = len(history) >= config.meta_rollback_warmup_steps
    exceeded_fraction = bool(
        warmed_up and fraction > config.meta_max_rollback_fraction)
    exceeded_consecutive = bool(
        consecutive_steps >= config.meta_max_consecutive_rollback_steps)
    return {
        "window_steps": len(recent),
        "window_decisions": decisions,
        "window_rollbacks": rollbacks,
        "window_rollback_fraction": fraction,
        "consecutive_rollback_steps": consecutive_steps,
        "warmed_up": warmed_up,
        "exceeded_fraction": exceeded_fraction,
        "exceeded_consecutive": exceeded_consecutive,
        "unhealthy": exceeded_fraction or exceeded_consecutive,
    }


def training_termination_record(
    ctx: RunContext, reason: str, *, horizon: int | None = None,
    health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "reason": reason,
        "actual_meta_steps": int(ctx.state["meta_step"]),
        "target_meta_steps": ctx.config.meta_max_steps,
        "best_meta_step": ctx.state.get("best_meta_step"),
        "best_dev_loss": ctx.state.get("best_dev_loss"),
    }
    if horizon is not None:
        record["horizon"] = int(horizon)
    if health is not None:
        record["health"] = dict(health)
    return record


def finalize_meta_training(
    ctx: RunContext, termination: Mapping[str, Any], *, health_bound: bool,
) -> None:
    """Durably select/freeze the best policy after success or a safe bound."""
    record = dict(termination)
    ctx.state["training_termination"] = record
    ctx.state["phase"] = "meta_training_finalizing"
    ctx.state["meta_finalization"] = {
        "status": "detected",
        "health_bound": health_bound,
        "termination": record,
    }
    if health_bound:
        ctx.state["early_stop"] = record
        ctx.event("meta_training_health_bound_detected", level="warning",
                  critical=True, termination=record)
        ctx.checkpoint("meta_training_health_bound", force=True)
    else:
        ctx.state["early_stop"] = None
        ctx.event("meta_training_complete", critical=True,
                  termination=record)
        ctx.checkpoint("meta_training_complete", force=True)
    ctx.freeze_best()
    ctx.state["meta_finalization"]["status"] = "frozen"
    ctx.state["phase"] = "evaluation"
    ctx.checkpoint("policy_frozen", include_policy=False, force=True)
    ctx.event(("meta_training_health_bound_finalized" if health_bound
               else "policy_frozen"), critical=True,
              best_dev_loss=ctx.state.get("best_dev_loss"),
              best_meta_step=ctx.state.get("best_meta_step"),
              termination=record)


def run_meta_training(ctx: RunContext) -> None:
    ctx.state["phase"] = "meta_training"
    ctx.ensure_policy()
    assert ctx.policy is not None and ctx.outer_optimizer is not None
    if int(ctx.state["meta_step"]) == 0:
        ctx.preserve_zero_shot()
        if not any(int(row["meta_step"]) == 0
                   for row in ctx.state["dev_history"]):
            dev = run_dev_evaluation(ctx)
            dev_row = {"meta_step": 0, **dev}
            ctx.state["dev_history"].append(dev_row)
            ctx.state["best_dev_loss"] = dev["validation_loss"]
            ctx.save_best(float(dev["validation_loss"]))
            ctx.event("dev_evaluation", critical=True, **dev_row)
            ctx.event("best_policy_updated", critical=True, **dev_row)
        ctx.checkpoint("meta_step_zero_controls", force=True)
    ctx.event("meta_training_start", critical=True,
              target_steps=ctx.config.meta_max_steps,
              train_specs=ctx.config.meta_train_specs)
    while int(ctx.state["meta_step"]) < ctx.config.meta_max_steps:
        step = int(ctx.state["meta_step"])
        slot_index = step % len(ctx.config.meta_train_specs)
        slot = ctx.state["meta_slots"][slot_index]
        if slot is None or int(slot["age"]) >= ctx.config.deployment_steps:
            episode = 0 if slot is None else int(slot["episode"]) + 1
            slot = initialize_slot(
                ctx.config, slot_index, episode, ctx.device,
                selected_learning_rate(
                    ctx.state, "policy_base",
                    ctx.config.meta_train_specs[slot_index]))
            ctx.event("meta_episode_reset", slot=slot_index,
                      spec=slot["spec"], episode=episode)
        peak = int(torch.cuda.max_memory_allocated(ctx.device))
        horizon = ctx.config.horizon(step, peak)
        slot, metrics = meta_outer_step(
            ctx.policy, ctx.outer_optimizer, slot, step,
            ctx.config, ctx.device, horizon)
        ctx.state["meta_slots"][slot_index] = slot
        ctx.state["meta_step"] = step + 1
        ctx.state["meta_history"].append(metrics)
        ctx.event("meta_step", **metrics)

        if int(metrics["rollback_count"]) > 0:
            health = meta_rollback_health(ctx.state["meta_history"], ctx.config)
            record = {"type": "meta_rollback", "metrics": metrics,
                      "health": health}
            ctx.state["alarms"].append(record)
            ctx.event("meta_rollback_warning", level="warning", critical=True,
                      metrics=metrics, health=health)
            if bool(health["unhealthy"]):
                reason = ("meta_rollback_fraction_health_bound"
                          if health["exceeded_fraction"]
                          else "meta_consecutive_rollback_health_bound")
                health_alarm = {
                    "type": "meta_rollback_health_bound",
                    "metrics": metrics,
                    "health": health,
                    "reason": reason,
                }
                ctx.state["alarms"].append(health_alarm)
                ctx.event("meta_rollback_rate_alarm", level="error", critical=True,
                          metrics=metrics, health=health, reason=reason)
                termination = training_termination_record(
                    ctx, reason, horizon=horizon, health=health)
                finalize_meta_training(ctx, termination, health_bound=True)
                return
            ctx.checkpoint("meta_rollback", force=True)
        production_safety = core.SafetyConfig(
            max_policy_residual_ratio=ctx.config.residual_ratio)
        if (float(metrics["min_layer_firing_rate"])
                < production_safety.min_spike_rate
                or float(metrics["max_layer_firing_rate"])
                > production_safety.max_spike_rate):
            alarm = {"type": "meta_firing_rate", "metrics": metrics}
            ctx.state["alarms"].append(alarm)
            ctx.event("meta_firing_rate_alarm", level="error", critical=True,
                      metrics=metrics)
            ctx.checkpoint("meta_firing_rate_alarm", force=True)
            raise HealthAbort("committed meta state violated per-layer firing limits")
        if int(metrics["fallback_count"]) > 0:
            ctx.event("meta_fallback_warning", level="warning", critical=True,
                      metrics=metrics)
        recent = ctx.state["meta_history"][-10:]
        if len(recent) == 10 and statistics.fmean(
                float(item["mean_reward"]) for item in recent) < 0:
            ctx.event("negative_reward_window", level="warning", critical=True,
                      window=10, metrics=recent[-1])

        if (step + 1) % ctx.config.dev_every == 0 or step + 1 == ctx.config.meta_max_steps:
            dev = run_dev_evaluation(ctx)
            dev_row = {"meta_step": step + 1, **dev}
            ctx.state["dev_history"].append(dev_row)
            ctx.event("dev_evaluation", critical=True, **dev_row)
            best = ctx.state.get("best_dev_loss")
            if best is None or dev["validation_loss"] < float(best):
                ctx.state["best_dev_loss"] = dev["validation_loss"]
                ctx.save_best(dev["validation_loss"])
                ctx.event("best_policy_updated", critical=True, **dev_row)

        ctx.checkpoint("meta_periodic")
        ctx.safe_point()

    termination = training_termination_record(ctx, "target_meta_steps_reached")
    finalize_meta_training(ctx, termination, health_bound=False)


def _result_exists(rows: Sequence[dict[str, Any]], axis: str, spec: str,
                   seed: int, method: str) -> bool:
    return any(row["axis"] == axis and row["spec"] == spec
               and row["task_seed"] == seed and row["method"] == method
               for row in rows)


def _is_cuda_oom(error: BaseException) -> bool:
    oom_types = tuple(
        error_type for error_type in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None))
        if isinstance(error_type, type))
    if oom_types and isinstance(error, oom_types):
        return True
    message = str(error).lower()
    return (isinstance(error, RuntimeError)
            and "cuda" in message and "out of memory" in message)


def _evaluate_architecture(
    ctx: RunContext, axis: str, spec: str, spec_index: int,
    methods: Sequence[str], rows: list[dict[str, Any]],
) -> None:
    """Evaluate one architecture; callers own the CUDA-OOM transaction."""
    snn = ctx.config.snn(spec)
    for repeat, seed in enumerate(ctx.config.eval_task_seeds):
        task = core.make_synthetic_task(ctx.config.task_config(), ctx.device, seed)
        init_seed = ctx.config.policy_seed * 1_000_003 + spec_index * 1009 + repeat
        method_seed = seed * 65_537 + spec_index
        for method in methods:
            if _result_exists(rows, axis, spec, seed, method):
                continue
            ctx.state["evaluation_cursor"] = {
                "axis": axis, "spec": spec, "task_seed": seed,
                "method": method}
            ctx.event("evaluation_method_start", axis=axis, spec=spec,
                      snn_parameters=snn.parameter_count,
                      task_seed=seed, method=method)
            if method == "untrained":
                parameters = core.initialize_snn(snn, ctx.device, init_seed)
                result = {
                    "method": method,
                    "validation": core.evaluate_snn(
                        parameters, task.validation, snn),
                    "test": core.evaluate_snn(parameters, task.test, snn),
                    "train_loss_auc": None, "loss_curve": [],
                    "accepted": 0, "fallbacks": 0, "rollbacks": 0,
                    "firing_violations": 0,
                    "min_layer_firing_rate": None,
                    "max_layer_firing_rate": None,
                    "max_voltage": None,
                    "min_post_layer_firing_rate": None,
                    "max_post_layer_firing_rate": None,
                    "max_post_voltage": None,
                    "elapsed_seconds": 0.0,
                }
            else:
                current = ctx.state.get("current_method")
                identity = (axis, spec, seed, method)
                if (current is None
                        or tuple(current.get("identity", ())) != identity):
                    current = new_method_state(
                        method, snn, ctx.device, init_seed, task,
                        ctx.config.batch_size, method_seed)
                    current["identity"] = identity
                while int(current["step"]) < ctx.config.deployment_steps:
                    method_policy = (ctx.zero_shot_policy
                                     if method == "zero_shot_unguarded"
                                     else ctx.policy)
                    current, progress = method_step(
                        current, method_policy, task, snn, ctx.config,
                        selected_learning_rate(
                            ctx.state,
                            ("bptt" if method in {"bptt", "stdp"}
                             else "policy_base"),
                            spec),
                        selected_learning_rate(ctx.state, "stdp", spec),
                        method_seed)
                    current["identity"] = identity
                    ctx.state["current_method"] = current
                    step = int(current["step"])
                    if (step % ctx.log_every == 0
                            or progress["decision"] != "accepted"
                            or bool(progress["firing_alarm"])):
                        ctx.event("evaluation_step", axis=axis, spec=spec,
                                  task_seed=seed, **progress)
                    if step % ctx.checkpoint_every == 0:
                        ctx.checkpoint("evaluation_progress",
                                       include_policy=False, force=True)
                    ctx.safe_point(include_policy=False)
                result = finalize_method(current, task, snn)
                ctx.state["current_method"] = None
            row = {"axis": axis, "spec": spec,
                   "snn_parameters": snn.parameter_count,
                   "task_seed": seed, **result}
            rows.append(row)
            ctx.event("evaluation_method_complete", critical=True,
                      axis=axis, spec=spec, task_seed=seed,
                      method=method, validation=row["validation"],
                      test=row["test"], train_loss_auc=row["train_loss_auc"],
                      fallbacks=row["fallbacks"], rollbacks=row["rollbacks"],
                      firing_violations=row.get("firing_violations", 0),
                      min_layer_firing_rate=row.get("min_layer_firing_rate"),
                      max_layer_firing_rate=row.get("max_layer_firing_rate"),
                      max_voltage=row.get("max_voltage"),
                      min_post_layer_firing_rate=row.get(
                          "min_post_layer_firing_rate"),
                      max_post_layer_firing_rate=row.get(
                          "max_post_layer_firing_rate"),
                      max_post_voltage=row.get("max_post_voltage"),
                      seconds=row["elapsed_seconds"])
            ctx.checkpoint("evaluation_method_complete",
                           include_policy=False, force=True)


def run_evaluation(ctx: RunContext) -> None:
    assert ctx.policy is not None
    ctx.ensure_zero_shot_policy()
    assert ctx.zero_shot_policy is not None
    ctx.state["phase"] = "evaluation"
    methods = ("untrained", "bptt", "stdp", "zero_shot_unguarded",
               "policy_unguarded", "policy_guarded")
    axes = (("width", ctx.config.width_specs), ("depth", ctx.config.depth_specs))
    rows = ctx.state["evaluation_results"]
    capacity_bounds = ctx.state.setdefault("capacity_bounds", {})
    for axis, specs in axes:
        if axis in capacity_bounds:
            ctx.event("evaluation_axis_capacity_skip", level="warning", critical=True,
                      axis=axis, capacity_bound=capacity_bounds[axis])
            continue
        for spec_index, spec in enumerate(specs):
            torch.cuda.reset_peak_memory_stats(ctx.device)
            capacity_record: dict[str, Any] | None = None
            try:
                _evaluate_architecture(ctx, axis, spec, spec_index, methods, rows)
            except BaseException as error:
                if not _is_cuda_oom(error):
                    raise
                snn = ctx.config.snn(spec)
                capacity_record = {
                    "kind": "cuda_oom",
                    "timestamp_utc": utc_timestamp(),
                    "axis": axis,
                    "spec": spec,
                    "spec_index": spec_index,
                    "snn_parameters": snn.parameter_count,
                    "cursor": dict(ctx.state.get("evaluation_cursor") or {}),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "gpu_at_oom": gpu_telemetry(ctx.device),
                }
                ctx.state["current_method"] = None

            if capacity_record is not None:
                # Outside the except block, its traceback no longer retains the
                # failed architecture's CUDA tensors.
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(ctx.device)
                capacity_record["gpu_after_cleanup"] = gpu_telemetry(ctx.device)
                capacity_bounds[axis] = capacity_record
                ctx.state["evaluation_cursor"] = {
                    "axis": axis, "status": "capacity_bound", "spec": spec}
                ctx.event("evaluation_capacity_bound", level="warning", critical=True,
                          **capacity_record)
                # Persist the axis skip marker immediately so resume cannot
                # retry the same impossible allocation forever.
                ctx.checkpoint("evaluation_capacity_bound",
                               include_policy=False, force=True)
                break

    summary = aggregate_results(
        rows, ctx.config, ctx.state.get("capacity_bounds"))
    ctx.state["summary"] = summary
    termination = (ctx.state.get("training_termination")
                   or ctx.state.get("early_stop"))
    if termination is None:
        termination = training_termination_record(
            ctx, ("target_meta_steps_reached"
                  if int(ctx.state["meta_step"]) >= ctx.config.meta_max_steps
                  else "incomplete_legacy_run"))
    ctx.state["training_termination"] = termination
    alarms = alarm_summary(ctx.state.get("alarms", ()))
    ctx.state["phase"] = "complete"
    ctx.state["status"] = "complete"
    atomic_write_json(ctx.run_dir / "results.json", {
        "schema": PRODUCTION_SCHEMA,
        "run_id": ctx.run_id,
        "source_sha256": ctx.source_sha256,
        "production_manifest_sha256": ctx.production_manifest_sha256,
        "production_verification": ctx.production_manifest,
        "config": dataclasses.asdict(ctx.config),
        "selected_lrs": ctx.state["selected_lrs"],
        "best_dev_loss": ctx.state.get("best_dev_loss"),
        "best_meta_step": ctx.state.get("best_meta_step"),
        "completed_meta_steps": int(ctx.state["meta_step"]),
        "target_meta_steps": ctx.config.meta_max_steps,
        "training_termination": termination,
        "early_stop": ctx.state.get("early_stop"),
        "alarm_summary": alarms,
        "zero_shot_policy_milestone": ctx.state.get("zero_shot_policy_milestone"),
        "task_controls": ctx.state.get("task_controls"),
        "meta_history": ctx.state["meta_history"],
        "dev_history": ctx.state["dev_history"],
        "individual_results": rows,
        "capacity_bounds": capacity_bounds,
        "summary": summary,
    })
    ctx.checkpoint("run_complete", include_policy=False, force=True)
    ctx.event("run_complete", critical=True, summary=summary,
              training_termination=termination, early_stop=ctx.state.get("early_stop"),
              alarm_summary=alarms,
              completed_meta_steps=int(ctx.state["meta_step"]),
              target_meta_steps=ctx.config.meta_max_steps)


def production_verification_identity(
    device: torch.device, core_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PRODUCTION_VERIFY_SCHEMA,
        "source_sha256": production_source_digest(),
        "core_manifest_sha256": stable_config_digest(core_manifest),
        "device": core.verification_identity(device),
    }


def verify_resume_equivalence(device: torch.device) -> None:
    config = ProductionConfig(
        train_samples=32, validation_samples=16, test_samples=16,
        batch_size=8, deployment_steps=8, meta_max_steps=2,
        meta_train_specs=("1x4",), width_specs=("1x4",), depth_specs=("2x4",),
        dev_task_seeds=(71,), eval_task_seeds=(81,), tune_steps=8,
        dev_every=2, dev_steps=4, residual_warmup=1)
    config.validate()
    tiny_config = core.PolicyConfig(channels=16, blocks=1, groups=4,
                                    sketch_bins=8, checkpoint_blocks=False)
    core.seed_everything(123)
    with torch.device(device):
        initial_policy = core.AlphaZeroPolicyOptimizer(
            tiny_config, enforce_floor=False)
    initial_state = {key: value.detach().clone()
                     for key, value in initial_policy.state_dict().items()}

    def build() -> tuple[core.AlphaZeroPolicyOptimizer, torch.optim.Optimizer]:
        with torch.device(device):
            policy = core.AlphaZeroPolicyOptimizer(tiny_config, enforce_floor=False)
        policy.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3, fused=True)
        return policy, optimizer

    continuous_policy, continuous_optimizer = build()
    continuous_slot = initialize_slot(config, 0, 0, device, 3e-3)
    continuous_slot, _ = meta_outer_step(
        continuous_policy, continuous_optimizer, continuous_slot, 0,
        config, device, 2)
    continuous_slot, _ = meta_outer_step(
        continuous_policy, continuous_optimizer, continuous_slot, 1,
        config, device, 2)

    resumed_policy, resumed_optimizer = build()
    resumed_slot = initialize_slot(config, 0, 0, device, 3e-3)
    resumed_slot, _ = meta_outer_step(
        resumed_policy, resumed_optimizer, resumed_slot, 0,
        config, device, 2)
    with tempfile.TemporaryDirectory(prefix="snn-prod-resume-") as directory:
        manager = CheckpointManager(Path(directory), keep_last=3)
        manager.save({"policy": resumed_policy.state_dict(),
                      "optimizer": resumed_optimizer.state_dict(),
                      "slot": resumed_slot}, config={"test": "resume"})
        loaded = manager.load_latest(
            map_location=device, expected_config={"test": "resume"}).payload
    resumed_policy, resumed_optimizer = build()
    resumed_policy.load_state_dict(loaded["policy"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    resumed_slot = loaded["slot"]
    resumed_slot, _ = meta_outer_step(
        resumed_policy, resumed_optimizer, resumed_slot, 1,
        config, device, 2)

    for left, right in zip(continuous_policy.parameters(), resumed_policy.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(continuous_slot["parameters"], resumed_slot["parameters"]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    left_state = unpack_optimizer_state(continuous_slot["optimizer_state"])
    right_state = unpack_optimizer_state(resumed_slot["optimizer_state"])
    if left_state.step != right_state.step:
        raise AssertionError("resumed optimizee step differs")
    for left, right in zip(left_state.momentum, right_state.momentum):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def verify_scientific_controls(device: torch.device) -> None:
    config = ProductionConfig(
        train_samples=32, validation_samples=16, test_samples=16,
        batch_size=8, deployment_steps=4, meta_max_steps=2,
        meta_train_specs=("1x4",), width_specs=("1x4",),
        depth_specs=("2x4",), dev_task_seeds=(171, 172),
        eval_task_seeds=(181,), tune_steps=8, dev_every=1,
        dev_steps=2, residual_warmup=1)
    config.validate()
    tiny_config = core.PolicyConfig(
        channels=16, blocks=1, groups=4, sketch_bins=8,
        checkpoint_blocks=False)
    core.seed_everything(321)
    with torch.device(device):
        policy = core.AlphaZeroPolicyOptimizer(tiny_config, enforce_floor=False)

    dev = evaluate_policy_on_dev_tasks(policy, config, device, 3e-3, 2e-3)
    per_seed = dev["per_seed"]
    if len(per_seed) != len(config.dev_task_seeds):
        raise AssertionError("dev selection did not cover every configured seed")
    if len({row["init_seed"] for row in per_seed}) != len(per_seed):
        raise AssertionError("dev selection reused an initialization seed")
    if len({row["method_seed"] for row in per_seed}) != len(per_seed):
        raise AssertionError("dev selection reused a method seed")
    expected_dev_loss = statistics.fmean(
        float(row["validation_loss"]) for row in per_seed)
    if dev["validation_loss"] != expected_dev_loss:
        raise AssertionError("dev selection loss is not the all-seed mean")

    task = core.make_synthetic_task(config.task_config(), device, 181)
    snn = config.snn("1x4")
    rate_state = {
        "selected_lrs": {
            "bptt": {"1x4": 3e-3},
            "policy_base": {"1x4": 1e-3},
            "stdp": {"1x4": 2e-3},
        }
    }
    if selected_learning_rate(rate_state, "policy_base", "1x4") != 1e-3:
        raise AssertionError("per-architecture learning-rate lookup failed")
    guarded_adam = new_method_state(
        "adam_guarded", snn, device, 777, task, 8, 888)
    guarded_adam, guarded_progress = method_step(
        guarded_adam, None, task, snn, config, 1e-3, 0.0, 888)
    if guarded_progress["method"] != "adam_guarded":
        raise AssertionError("policy-free guarded Adam control did not execute")
    bptt = new_method_state("bptt", snn, device, 777, task, 8, 888)
    zero = new_method_state(
        "zero_shot_unguarded", snn, device, 777, task, 8, 888)
    for _ in range(3):
        bptt, bptt_progress = method_step(
            bptt, policy, task, snn, config, 3e-3, 2e-3, 888)
        zero, zero_progress = method_step(
            zero, policy, task, snn, config, 3e-3, 2e-3, 888)
        for progress in (bptt_progress, zero_progress):
            if not {"min_layer_firing_rate", "max_layer_firing_rate",
                    "firing_alarm"}.issubset(progress):
                raise AssertionError("evaluation omitted per-layer firing telemetry")
        for key in ("loss", "firing_rate", "min_layer_firing_rate",
                    "max_layer_firing_rate", "max_voltage"):
            if bptt_progress[key] != zero_progress[key]:
                raise AssertionError(
                    f"zero-shot/BPTT pre-update telemetry differs for {key}")
    for left, right in zip(bptt["parameters"], zero["parameters"]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    with tempfile.TemporaryDirectory(prefix="snn-prod-zero-shot-") as directory:
        manager = CheckpointManager(
            Path(directory), prefix="zero-shot-policy", keep_last=1)
        original = policy.actor_head[-1].bias.detach().cpu().clone()
        manager.save({"policy": policy.state_dict(), "meta_step": 0},
                     config={"kind": "zero_shot"})
        with torch.no_grad():
            policy.actor_head[-1].bias.add_(1.0)
        loaded = manager.load_latest(
            expected_config={"kind": "zero_shot"}).payload
        torch.testing.assert_close(
            loaded["policy"]["actor_head.3.bias"], original, rtol=0, atol=0)
        if loaded["meta_step"] != 0:
            raise AssertionError("zero-shot snapshot lost its meta-step identity")

    # Exercise best-policy selection metadata without constructing a full run
    # context or allocating the production-sized policy.
    core.seed_everything(654)
    with torch.device(device):
        best_policy = core.AlphaZeroPolicyOptimizer(
            tiny_config, enforce_floor=False)
    with tempfile.TemporaryDirectory(prefix="snn-prod-best-") as directory:
        fake = object.__new__(RunContext)
        fake.device = device
        fake.policy = best_policy
        fake.outer_optimizer = None
        fake.checkpoint_config = {"kind": "best-selection"}
        fake.state = {"meta_step": 0, "best_dev_loss": 1.25,
                      "best_policy_milestone": None, "best_meta_step": None}
        fake.best_checkpoints = CheckpointManager(
            Path(directory) / "best", prefix="best-policy", keep_last=1)
        fake.frozen_checkpoints = CheckpointManager(
            Path(directory) / "frozen", prefix="frozen-policy", keep_last=1)
        RunContext.save_best(fake, 1.25)
        with torch.no_grad():
            best_policy.actor_head[-1].bias.add_(1.0)
        fake.state["meta_step"] = 2
        RunContext.freeze_best(fake)
        frozen = fake.frozen_checkpoints.load_latest(
            expected_config=fake.checkpoint_config).payload
        if frozen["selected_meta_step"] != 0 or fake.state["best_meta_step"] != 0:
            raise AssertionError("frozen policy lost the selected best meta step")
        if frozen["best_dev_loss"] != 1.25:
            raise AssertionError("frozen policy lost the selected dev loss")


def run_production_verification(
    device: torch.device, core_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    core_manifest = core.require_verification_manifest(device, core_manifest_path)
    output_path.unlink(missing_ok=True)
    source_before = production_source_digest()
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Any) -> None:
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        row = {"name": name, "passed": True,
               "seconds": time.perf_counter() - started}
        checks.append(row)
        print(f"  PASS {name} ({row['seconds']:.3f}s)", flush=True)

    from snn_production_support import self_test as support_self_test
    with tempfile.TemporaryDirectory(prefix="snn-prod-verify-") as directory:
        check("durable_checkpoint_log_lock_signal_cpu",
              lambda: support_self_test(directory, "cpu"))
    with tempfile.TemporaryDirectory(prefix="snn-prod-verify-gpu-") as directory:
        check("durable_checkpoint_gpu_roundtrip",
              lambda: support_self_test(directory, device))

    def aggregation_check() -> None:
        small_sample = mean_ci95([1.0, 2.0, 3.0])
        expected_ci = 2.4841377118949786
        if not math.isclose(float(small_sample["ci95"]), expected_ci,
                            rel_tol=0, abs_tol=1e-12):
            raise AssertionError("small-sample CI did not use Student-t")
        rows = []
        for method in ("untrained", "bptt", "stdp", "zero_shot_unguarded",
                       "policy_unguarded", "policy_guarded"):
            rows.append({
                "axis": "width", "spec": "1x4", "method": method,
                "task_seed": 123,
                "snn_parameters": 88,
                "validation": {"loss": 1.0, "accuracy": 0.5},
                "test": {"loss": 1.1, "accuracy": 0.5},
                "train_loss_auc": None if method == "untrained" else 1.2,
                "elapsed_seconds": 0.1, "accepted": 1,
                "fallbacks": 0, "rollbacks": 0,
            })
        aggregation_config = dataclasses.replace(
            ProductionConfig(), stop_gap=0.17,
            stop_rejection_rate=0.23, stop_patience=3,
            learnability_improvement=0.07)
        capacity = {"width": {"kind": "cuda_oom", "spec": "2x8192"}}
        result = aggregate_results(rows, aggregation_config, capacity)
        if result["bound_thresholds"] != {
                "loss_gap": 0.17,
                "guard_rejection_rate": 0.23,
                "consecutive_sizes": 3,
                "learnability_improvement": 0.07}:
            raise AssertionError("aggregation ignored configured bound thresholds")
        if result["bounds"]["width"].get("capacity_bound") != capacity["width"]:
            raise AssertionError("aggregation dropped the CUDA capacity bound")
        if len(result["paired_effects"]) != 4:
            raise AssertionError("aggregation omitted paired method effects")
        if not _is_cuda_oom(torch.OutOfMemoryError("CUDA out of memory")):
            raise AssertionError("typed CUDA OOM was not recognized")
        if _is_cuda_oom(RuntimeError("unrelated runtime failure")):
            raise AssertionError("non-OOM runtime failure was misclassified")
        if alarm_summary([{"type": "rollback"}, {"type": "rollback"}]) != {
                "total": 2, "by_type": {"rollback": 2},
                "latest_type": "rollback"}:
            raise AssertionError("alarm summary is incomplete")
        with tempfile.TemporaryDirectory(prefix="snn-terminal-status-") as directory:
            status_path = Path(directory) / "status.json"
            atomic_write_json(status_path, {
                "run_id": "verified-run", "status": "error",
                "data": {"error_type": "InjectedCheckpointFailure"},
            })
            checkpoint_state = {"status": "running"}
            reconcile_terminal_status(
                checkpoint_state, "verified-run", status_path)
            if (checkpoint_state["status"] != "error"
                    or checkpoint_state["terminal_error"]["error_type"]
                    != "InjectedCheckpointFailure"):
                raise AssertionError(
                    "lightweight terminal marker did not block unsafe resume")
        json.dumps(result, allow_nan=False)

    def safe_objective_check() -> None:
        rejected_candidate = torch.tensor(2.0, device=device, requires_grad=True)
        committed = torch.tensor(1.0, device=device)
        rejected = safe_quality_loss(
            rejected_candidate, committed, "adam_fallback:firing_rate")
        (rejected + 0.0 * rejected_candidate).backward()
        torch.testing.assert_close(
            rejected_candidate.grad, torch.zeros_like(rejected_candidate))

        accepted_candidate = torch.tensor(2.0, device=device, requires_grad=True)
        accepted = safe_quality_loss(accepted_candidate, committed, "accepted")
        accepted.backward()
        torch.testing.assert_close(
            accepted_candidate.grad, torch.ones_like(accepted_candidate))

        health_config = dataclasses.replace(
            ProductionConfig(), meta_rollback_window=50,
            meta_rollback_warmup_steps=25,
            meta_max_rollback_fraction=0.10,
            meta_max_consecutive_rollback_steps=3)
        isolated = [{"horizon": 2, "rollback_count": 1}]
        if meta_rollback_health(isolated, health_config)["unhealthy"]:
            raise AssertionError("an isolated safe rollback aborted during warmup")
        consecutive = isolated * 3
        if not meta_rollback_health(consecutive, health_config)["unhealthy"]:
            raise AssertionError("consecutive rollback health alarm did not fire")
        frequent = [
            {"horizon": 2, "rollback_count": int(index % 2 == 0)}
            for index in range(25)
        ]
        if not meta_rollback_health(frequent, health_config)["exceeded_fraction"]:
            raise AssertionError("rollback-rate health alarm did not fire")

        class FinalizationProbe:
            def __init__(self) -> None:
                self.config = dataclasses.replace(ProductionConfig(), meta_max_steps=20)
                self.state = {
                    "meta_step": 7, "best_meta_step": 5,
                    "best_dev_loss": 0.25, "training_termination": None,
                    "early_stop": None,
                }
                self.events: list[tuple[str, dict[str, Any]]] = []
                self.checkpoints: list[tuple[str, bool]] = []
                self.snapshots: list[dict[str, Any]] = []
                self.crash_after_checkpoint: int | None = None
                self.frozen = False

            def event(self, name: str, **data: Any) -> None:
                self.events.append((name, data))

            def checkpoint(self, reason: str, *, include_policy: bool = True,
                           force: bool = False) -> None:
                self.checkpoints.append((reason, include_policy))
                self.snapshots.append(json.loads(json.dumps(self.state)))
                if self.crash_after_checkpoint == len(self.checkpoints):
                    raise RuntimeError("injected finalization interruption")

            def freeze_best(self) -> None:
                self.frozen = True

        probe = FinalizationProbe()
        termination = training_termination_record(
            probe, "meta_consecutive_rollback_health_bound", horizon=16,
            health={"unhealthy": True})
        finalize_meta_training(probe, termination, health_bound=True)
        if (probe.state["phase"] != "evaluation" or not probe.frozen
                or probe.state["early_stop"] != termination
                or probe.state["training_termination"] != termination):
            raise AssertionError("health-bound finalization was not durable")
        if [row[0] for row in probe.checkpoints] != [
                "meta_training_health_bound", "policy_frozen"]:
            raise AssertionError("health-bound finalization omitted a checkpoint")
        if [row[0] for row in probe.events] != [
                "meta_training_health_bound_detected",
                "meta_training_health_bound_finalized"]:
            raise AssertionError("health-bound finalization omitted an event")
        if probe.snapshots[0]["phase"] != "meta_training_finalizing":
            raise AssertionError("health-bound checkpoint can resume meta-training")
        if probe.snapshots[-1]["meta_finalization"]["status"] != "frozen":
            raise AssertionError("frozen transition was not durably identified")

        interrupted = FinalizationProbe()
        interrupted.crash_after_checkpoint = 1
        interrupted_termination = training_termination_record(
            interrupted, "meta_consecutive_rollback_health_bound", horizon=16,
            health={"unhealthy": True})
        try:
            finalize_meta_training(
                interrupted, interrupted_termination, health_bound=True)
        except RuntimeError as error:
            if str(error) != "injected finalization interruption":
                raise
        else:
            raise AssertionError("finalization interruption was not injected")
        resumed = FinalizationProbe()
        resumed.state = interrupted.snapshots[-1]
        finalize_meta_training(
            resumed, resumed.state["training_termination"], health_bound=True)
        if (resumed.state["phase"] != "evaluation" or not resumed.frozen
                or resumed.snapshots[0]["phase"] != "meta_training_finalizing"):
            raise AssertionError("interrupted health finalization did not resume safely")

    check("strict_result_aggregation", aggregation_check)
    check("rejected_action_gradient_isolation", safe_objective_check)
    check("zero_shot_and_all_seed_dev_controls",
          lambda: verify_scientific_controls(device))
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        check("interrupted_resume_exact_meta_trajectory",
              lambda: verify_resume_equivalence(device))
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
    identity = production_verification_identity(device, core_manifest)
    if identity["source_sha256"] != source_before:
        raise RuntimeError("production source changed during verification")
    manifest = {**identity, "passed": True, "checks": checks,
                "created_utc": utc_timestamp()}
    atomic_write_json(output_path, manifest)
    print(f"production verification passed: {len(checks)}/{len(checks)}")
    print(f"manifest: {output_path}")
    return manifest


def require_production_verification(
    path: Path, device: torch.device,
    core_manifest: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("production verification manifest is missing; run verify")
    manifest = json.loads(path.read_text())
    expected = production_verification_identity(device, core_manifest)
    if manifest.get("passed") is not True:
        raise RuntimeError("production verification did not pass")
    mismatches = [key for key, value in expected.items()
                  if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"production verification is stale/incompatible: {', '.join(mismatches)}")
    required = {
        "durable_checkpoint_log_lock_signal_cpu",
        "durable_checkpoint_gpu_roundtrip",
        "strict_result_aggregation",
        "rejected_action_gradient_isolation",
        "zero_shot_and_all_seed_dev_controls",
        "interrupted_resume_exact_meta_trajectory",
    }
    checks = manifest.get("checks")
    if not isinstance(checks, list) or {
            row.get("name") for row in checks if row.get("passed") is True} != required:
        raise RuntimeError("production verification check set is incomplete")
    return manifest


def run_experiment(args: argparse.Namespace, device: torch.device) -> int:
    core_manifest_path = Path(args.core_manifest)
    core_manifest = core.require_verification_manifest(device, core_manifest_path)
    production_manifest = require_production_verification(
        Path(args.production_manifest), device, core_manifest)
    config = config_from_args(args)
    run_dir = Path(args.run_dir)
    lock = RunLock(run_dir / "run.lock", run_id=None)
    with lock:
        ctx = RunContext(
            run_dir, device, config, core_manifest, production_manifest,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            heartbeat_seconds=args.heartbeat_seconds,
            log_every=args.log_every)
        lock.set_run_id(ctx.run_id)
        try:
            with ctx.signal:
                if ctx.state["status"] == "complete":
                    ctx.event("resume_complete_noop", critical=True)
                    return 0
                if ctx.state["status"] == "error":
                    ctx.event("resume_terminal_error_blocked", level="error",
                              critical=True,
                              terminal_error=ctx.state.get("terminal_error"))
                    return 2
                if ctx.state["phase"] == "task_controls":
                    run_task_controls(ctx)
                if ctx.state["phase"] == "tuning":
                    run_lr_tuning(ctx)
                if ctx.state["phase"] == "meta_training":
                    run_meta_training(ctx)
                if ctx.state["phase"] == "meta_training_finalizing":
                    termination = (ctx.state.get("training_termination")
                                   or ctx.state.get("early_stop"))
                    if termination is None:
                        raise HealthAbort(
                            "meta-training finalization has no termination record")
                    finalize_meta_training(
                        ctx, termination,
                        health_bound=ctx.state.get("early_stop") is not None)
                if ctx.state["phase"] == "evaluation":
                    run_evaluation(ctx)
        except GracefulStop:
            return 130
        except BaseException as error:
            failed_phase = ctx.state.get("phase")
            ctx.state["status"] = "error"
            terminal_error = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "failed_phase": failed_phase,
                "meta_step": ctx.state.get("meta_step"),
            }
            ctx.state["terminal_error"] = terminal_error
            ctx.logger.exception(
                "run_exception", error, phase=ctx.state.get("phase"),
                meta_step=ctx.state.get("meta_step"), gpu=gpu_telemetry(device))
            status_error = {
                **terminal_error,
                "traceback": "".join(traceback.format_exception(error)),
            }
            # Persist the lightweight terminal marker before the heavyweight
            # CUDA synchronization/checkpoint path, which may itself fail.
            ctx._write_status("run_exception", "error", status_error)
            try:
                ctx.checkpoint("run_exception", force=True)
            except BaseException as checkpoint_error:
                ctx.logger.exception(
                    "run_exception_checkpoint_failed", checkpoint_error,
                    original_error=terminal_error, gpu=gpu_telemetry(device))
            ctx._write_status("run_exception", "error", status_error)
            raise
        finally:
            ctx.close()
    return 0


def print_status(run_dir: Path) -> int:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        print(f"no status file in {run_dir}", file=sys.stderr)
        return 2
    print(status_path.read_text(), end="")
    results = run_dir / "results.json"
    if results.is_file():
        print(f"results: {results}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="verify durability and exact resume")
    verify.add_argument("--device", dest="device", default=argparse.SUPPRESS)
    verify.add_argument("--core-manifest", default="build/snn_meta_verification.json")
    verify.add_argument("--output", default="build/snn_production_verification.json")

    run = commands.add_parser("run", help="run or resume the production protocol")
    run.add_argument("--device", dest="device", default=argparse.SUPPRESS)
    run.add_argument("--core-manifest", default="build/snn_meta_verification.json")
    run.add_argument("--production-manifest",
                     default="build/snn_production_verification.json")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--checkpoint-every", type=int, default=25,
                     help="meta/evaluation steps between durable checkpoints")
    run.add_argument("--heartbeat-seconds", type=float, default=30.0)
    run.add_argument("--log-every", type=int, default=10)
    run.add_argument("--train-samples", type=int, default=4096)
    run.add_argument("--validation-samples", type=int, default=2048)
    run.add_argument("--test-samples", type=int, default=4096)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--deployment-steps", type=int, default=256)
    run.add_argument("--meta-max-steps", type=int, default=3000)
    run.add_argument("--meta-train-specs",
                     default="1x64,1x256,2x128,4x128,2x256,4x256")
    run.add_argument("--width-specs",
                     default="2x32,2x64,2x128,2x256,2x512")
    run.add_argument("--depth-specs",
                     default="1x128,2x128,4x128,8x128")
    run.add_argument("--dev-task-seeds", default="700001,700002,700003")
    run.add_argument("--eval-task-seeds", default="900001,900002,900003")
    run.add_argument("--tune-steps", type=int, default=128)
    run.add_argument("--dev-every", type=int, default=50)
    run.add_argument("--dev-steps", type=int, default=256)
    run.add_argument("--meta-seed", type=int, default=100001)
    run.add_argument("--policy-seed", type=int, default=1)
    run.add_argument("--meta-lr", type=float, default=2e-4)
    run.add_argument("--learnability-improvement", type=float, default=0.05,
                     help="minimum BPTT loss improvement over untrained for a learnable SNN")

    status = commands.add_parser("status", help="print a run's durable status")
    status.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "status":
        return print_status(Path(args.run_dir))
    try:
        device = core.require_cuda(args.device)
        if args.command == "verify":
            run_production_verification(
                device, Path(args.core_manifest), Path(args.output))
            return 0
        if min(args.checkpoint_every, args.log_every) < 1:
            raise ValueError("checkpoint/log intervals must be positive")
        if args.heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        return run_experiment(args, device)
    except KeyboardInterrupt:
        return 130
    except BaseException as error:
        print(f"snn_production: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
