#!/usr/bin/env python3
"""Regenerate the corrected production reanalysis from raw run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snn_production as production
from snn_production_support import atomic_write_json


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stats(values: list[float]) -> dict[str, Any]:
    result = production.mean_ci95(values)
    ci = result["ci95"]
    result["lower"] = None if ci is None else float(result["mean"]) - float(ci)
    result["upper"] = None if ci is None else float(result["mean"]) + float(ci)
    return result


def build_reanalysis(run_dir: Path) -> dict[str, Any]:
    raw_bytes = (run_dir / "results.json").read_bytes()
    event_bytes = (run_dir / "events.jsonl").read_bytes()
    termination_bytes = (run_dir / "termination.json").read_bytes()
    raw = json.loads(raw_bytes)
    termination = json.loads(termination_bytes)
    events = [json.loads(line) for line in event_bytes.splitlines()]
    rows = raw["individual_results"]
    if termination["run_id"] != raw["run_id"]:
        raise RuntimeError("termination and results run IDs differ")
    if termination["results_sha256"] != sha256(raw_bytes):
        raise RuntimeError("termination record does not match results.json")
    if termination["events_sha256"] != sha256(event_bytes):
        raise RuntimeError("termination record does not match events.jsonl")

    def method_rows(axis: str, spec: str, method: str) -> list[dict[str, Any]]:
        return sorted(
            (row for row in rows
             if row["axis"] == axis and row["spec"] == spec
             and row["method"] == method),
            key=lambda row: row["task_seed"],
        )

    architectures = []
    axis_specs = [("width", spec) for spec in raw["config"]["width_specs"]]
    axis_specs += [("depth", spec) for spec in raw["config"]["depth_specs"]]
    for axis, spec in axis_specs:
        methods: dict[str, Any] = {}
        for method in ("stdp", "bptt", "zero_shot_unguarded",
                       "policy_unguarded", "policy_guarded"):
            items = method_rows(axis, spec, method)
            if len(items) != len(raw["config"]["eval_task_seeds"]):
                raise RuntimeError(f"incomplete {axis}/{spec}/{method} rows")
            methods[method] = {
                "test_loss": stats([float(row["test"]["loss"]) for row in items]),
                "test_accuracy": stats(
                    [float(row["test"]["accuracy"]) for row in items]),
            }
            if method == "policy_guarded":
                methods[method]["rejection_rate"] = stats([
                    (int(row["fallbacks"]) + int(row["rollbacks"]))
                    / max(1, int(row["accepted"]) + int(row["fallbacks"])
                          + int(row["rollbacks"]))
                    for row in items
                ])
        meta = method_rows(axis, spec, "policy_unguarded")
        zero = method_rows(axis, spec, "zero_shot_unguarded")
        if [row["task_seed"] for row in meta] != [row["task_seed"] for row in zero]:
            raise RuntimeError("paired task seeds differ")
        architectures.append({
            "axis": axis,
            "spec": spec,
            "snn_parameters": int(meta[0]["snn_parameters"]),
            "methods": methods,
            "paired_effects": {
                "meta_minus_zero_test_loss": stats([
                    float(left["test"]["loss"]) - float(right["test"]["loss"])
                    for left, right in zip(meta, zero)
                ]),
                "meta_minus_zero_test_accuracy": stats([
                    float(left["test"]["accuracy"])
                    - float(right["test"]["accuracy"])
                    for left, right in zip(meta, zero)
                ]),
            },
        })

    def architecture_average(method_a: str, method_b: str, metric: str) -> dict[str, Any]:
        differences = []
        for seed in raw["config"]["eval_task_seeds"]:
            left = [float(row["test"][metric]) for row in rows
                    if row["task_seed"] == seed and row["method"] == method_a]
            right = [float(row["test"][metric]) for row in rows
                     if row["task_seed"] == seed and row["method"] == method_b]
            if len(left) != 9 or len(right) != 9:
                raise RuntimeError("architecture-average comparison is incomplete")
            differences.append(statistics.fmean(left) - statistics.fmean(right))
        return stats(differences)

    horizons: dict[str, dict[str, Any]] = {}
    for row in raw["meta_history"]:
        key = str(row["horizon"])
        horizons.setdefault(key, {"outer_steps": 0, "decisions": 0, "rollbacks": 0})
        horizons[key]["outer_steps"] += 1
        horizons[key]["decisions"] += int(row["horizon"])
        horizons[key]["rollbacks"] += int(row["rollback_count"])
    for value in horizons.values():
        value["rollback_rate"] = value["rollbacks"] / value["decisions"]

    guard_rows = [row for row in rows if row["method"] == "policy_guarded"]
    guard = {name: sum(int(row[name]) for row in guard_rows)
             for name in ("accepted", "fallbacks", "rollbacks", "firing_violations")}
    guard["updates"] = guard["accepted"] + guard["fallbacks"] + guard["rollbacks"]
    guard["rejection_rate"] = (
        guard["fallbacks"] + guard["rollbacks"]) / guard["updates"]

    gpu_rows = [event.get("data", {}).get("gpu") for event in events
                if isinstance(event.get("data", {}).get("gpu"), dict)]
    checkpoints = [event for event in events if event["event"] == "checkpoint_complete"]
    levels: dict[str, int] = {}
    for event in events:
        levels[event["level"]] = levels.get(event["level"], 0) + 1

    return {
        "schema": "snn.production.reanalysis.v1",
        "run_id": raw["run_id"],
        "raw_source_sha256": raw["source_sha256"],
        "analysis_source_sha256": production.production_source_digest(),
        "raw_artifacts": {
            "results_sha256": sha256(raw_bytes),
            "events_sha256": sha256(event_bytes),
            "termination_sha256": sha256(termination_bytes),
        },
        "confidence_intervals": {
            "method": "two_sided_student_t_95_percent",
            "evaluation_task_seeds": len(raw["config"]["eval_task_seeds"]),
            "degrees_of_freedom": len(raw["config"]["eval_task_seeds"]) - 1,
            "multiple_comparison_correction": False,
        },
        "training_termination": termination,
        "best_checkpoint": {
            "meta_step": raw["best_meta_step"],
            "development_loss": raw["best_dev_loss"],
        },
        "selected_learning_rates": raw["selected_lrs"],
        "architecture_results": architectures,
        "architecture_averaged_paired_test_effects": {
            "meta_minus_zero_loss": architecture_average(
                "policy_unguarded", "zero_shot_unguarded", "loss"),
            "meta_minus_zero_accuracy": architecture_average(
                "policy_unguarded", "zero_shot_unguarded", "accuracy"),
            "meta_minus_bptt_loss": architecture_average(
                "policy_unguarded", "bptt", "loss"),
            "meta_minus_bptt_accuracy": architecture_average(
                "policy_unguarded", "bptt", "accuracy"),
            "meta_minus_stdp_loss": architecture_average(
                "policy_unguarded", "stdp", "loss"),
            "meta_minus_stdp_accuracy": architecture_average(
                "policy_unguarded", "stdp", "accuracy"),
        },
        "meta_training_rollback_by_horizon": horizons,
        "deployment_guard": guard,
        "bounds": {
            "architecture_capacity_bounds": raw["capacity_bounds"],
            "aggregate_architecture_bounds": raw["summary"]["bounds"],
            "largest_tested_width": {"spec": "2x512", "snn_parameters": 273412},
            "largest_tested_depth": {"spec": "8x128", "snn_parameters": 118276},
            "meta_training_horizon_health_bound": termination,
        },
        "operations": {
            "event_count": len(events),
            "event_levels": levels,
            "checkpoint_count": len(checkpoints),
            "checkpoint_seconds": sum(
                float(event["data"].get("seconds", 0.0)) for event in checkpoints),
            "checkpoint_bytes_written": sum(
                int(event["data"].get("size_bytes", 0)) for event in checkpoints),
            "gpu": {
                "peak_allocated_bytes": max(
                    int(gpu["max_allocated_bytes"]) for gpu in gpu_rows),
                "peak_reserved_bytes": max(
                    int(gpu["max_reserved_bytes"]) for gpu in gpu_rows),
                "minimum_free_bytes": min(int(gpu["free_bytes"]) for gpu in gpu_rows),
                "maximum_temperature_c": max(
                    int(gpu["temperature_c"]) for gpu in gpu_rows
                    if gpu.get("temperature_c") is not None),
                "maximum_utilization_percent": max(
                    int(gpu["utilization_percent"]) for gpu in gpu_rows
                    if gpu.get("utilization_percent") is not None),
            },
        },
        "limitations": [
            "one independent policy-training seed",
            "three task seeds measure within-policy rather than between-policy uncertainty",
            "no multiple-comparison correction",
            "accuracy is ceiling-saturated; cross-entropy is the primary endpoint",
            "K=16 meta-training stopped at its rollback-health bound before 3000 steps",
            "configured architecture ladders establish lower bounds only",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("build/runs/production-seed1-v2"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.run_dir / "reanalysis.json"
    atomic_write_json(output, build_reanalysis(args.run_dir))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
