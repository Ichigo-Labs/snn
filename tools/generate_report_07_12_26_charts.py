#!/usr/bin/env python3
"""Regenerate the figures used by docs/report_07-12-26.md."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("stdp", "STDP", "#7f7f7f", "o"),
    ("bptt", "surrogate BPTT", "#1f77b4", "s"),
    ("zero_shot_unguarded", "zero-shot policy", "#ff7f0e", "^"),
    ("policy_unguarded", "meta-policy", "#2ca02c", "D"),
    ("policy_guarded", "guarded meta-policy", "#9467bd", "P"),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "svg.hashsalt": "snn-report-2026-07-12",
    })


def save(fig: mpl.figure.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight",
                metadata={"Date": None, "Creator": "generate_report_07_12_26_charts.py"})
    plt.close(fig)


def architecture_labels(analysis: dict[str, Any]) -> list[str]:
    return [
        ("W " if row["axis"] == "width" else "D ") + row["spec"]
        for row in analysis["architecture_results"]
    ]


def benchmark_plot(analysis: dict[str, Any], output: Path, metric: str) -> None:
    rows = analysis["architecture_results"]
    labels = architecture_labels(analysis)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    key = "test_loss" if metric == "loss" else "test_accuracy"
    for method, name, color, marker in METHODS:
        means = np.array([row["methods"][method][key]["mean"] for row in rows])
        cis = np.array([row["methods"][method][key]["ci95"] for row in rows])
        if metric == "accuracy":
            means *= 100.0
            cis *= 100.0
        # Loss intervals are reported in the forest/table. Several df=2 lower
        # endpoints cross zero and therefore cannot be rendered on a log axis.
        yerr = None if metric == "loss" else cis
        ax.errorbar(x, means, yerr=yerr, label=name, color=color, marker=marker,
                    linewidth=1.7, markersize=5, capsize=2.5)
    ax.axvline(4.5, color="#999999", linewidth=1, linestyle=":")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    if metric == "loss":
        ax.set_yscale("log")
        ax.set_ylabel("test cross-entropy (log scale)")
        ax.set_title("Held-out test loss by SNN architecture (mean over n=3 task seeds)")
    else:
        ax.set_ylabel("test accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Held-out test accuracy by SNN architecture (mean ± 95% t interval, n=3)")
    ax.set_xlabel("architecture (W = width ladder, D = depth ladder)")
    ax.legend(ncol=3, loc="best")
    save(fig, output)


def paired_effect_plot(analysis: dict[str, Any], output: Path) -> None:
    rows = analysis["architecture_results"]
    labels = architecture_labels(analysis)
    means = [row["paired_effects"]["meta_minus_zero_test_loss"]["mean"] for row in rows]
    cis = [row["paired_effects"]["meta_minus_zero_test_loss"]["ci95"] for row in rows]
    overall = analysis["architecture_averaged_paired_test_effects"]["meta_minus_zero_loss"]
    labels.append("architecture average")
    means.append(overall["mean"])
    cis.append(overall["ci95"])
    y = np.arange(len(labels))
    colors = ["#2ca02c" if mean < 0 else "#d62728" for mean in means]
    colors[-1] = "#111111"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axvline(0.0, color="#555555", linewidth=1, linestyle="--")
    for index, (mean, ci, color) in enumerate(zip(means, cis, colors)):
        ax.errorbar(mean, index, xerr=ci, color=color, marker="o", capsize=3,
                    linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("paired Δ test CE: meta-policy − zero-shot (lower favors meta-policy)")
    ax.set_title("Meta-learning attribution effect with paired 95% Student-t intervals")
    save(fig, output)


def development_plot(results: dict[str, Any], output: Path) -> None:
    rows = results["dev_history"]
    steps = np.array([row["meta_step"] for row in rows])
    losses = np.array([row["validation_loss"] for row in rows])
    accuracy = np.array([row["validation_accuracy"] * 100.0 for row in rows])
    best_index = int(np.argmin(losses))
    fig, left = plt.subplots(figsize=(10.2, 4.8))
    spans = ((0, 250, "K=2", "#e8f1fa"), (250, 750, "K=4", "#fef0df"),
             (750, 2250, "K=8", "#e8f5e9"), (2250, 2502, "K=16", "#fce8e6"))
    for start, end, label, color in spans:
        left.axvspan(start, end, color=color, alpha=0.8)
        left.text((start + end) / 2, 0.98, label, transform=left.get_xaxis_transform(),
                  ha="center", va="top", fontsize=8)
    left.plot(steps, losses, color="#1f77b4", marker="o", markersize=3,
              linewidth=1.3, label="development CE")
    left.scatter([steps[best_index]], [losses[best_index]], color="#d62728", s=42,
                 zorder=4, label=f"selected best: step {steps[best_index]}")
    left.set_xlabel("meta-training outer step")
    left.set_ylabel("development cross-entropy", color="#1f77b4")
    left.tick_params(axis="y", labelcolor="#1f77b4")
    right = left.twinx()
    right.plot(steps, accuracy, color="#9467bd", linewidth=1.1, alpha=0.75,
               label="development accuracy")
    right.set_ylabel("development accuracy (%)", color="#9467bd")
    right.tick_params(axis="y", labelcolor="#9467bd")
    left.set_title("Development selection curve and curriculum horizons")
    handles1, labels1 = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    left.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, output)


def rollback_plot(analysis: dict[str, Any], output: Path) -> None:
    horizon_rows = analysis["meta_training_rollback_by_horizon"]
    horizons = sorted(horizon_rows, key=int)
    rates = [100.0 * horizon_rows[key]["rollback_rate"] for key in horizons]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    bars = ax.bar([f"K={key}" for key in horizons], rates,
                  color=["#9ecae1", "#6baed6", "#3182bd", "#de2d26"])
    for bar, key in zip(bars, horizons):
        row = horizon_rows[key]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{row["rollbacks"]}/{row["decisions"]}', ha="center", va="bottom")
    ax.axhline(10.0, color="#555555", linestyle="--", linewidth=1,
               label="configured 50-step rollback limit (10%)")
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("rollback decisions (%)")
    ax.set_title("Meta-training rollback rate increased with unroll horizon")
    ax.legend(loc="upper left")
    save(fig, output)


def guard_plot(analysis: dict[str, Any], output: Path) -> None:
    rows = analysis["architecture_results"]
    labels = architecture_labels(analysis)
    rates = [100.0 * row["methods"]["policy_guarded"]["rejection_rate"]["mean"]
             for row in rows]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    bars = ax.bar(np.arange(len(rows)), rates, color="#9467bd")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.12, f"{rate:.2f}%",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(25.0, color="#d62728", linestyle="--", linewidth=1.2,
               label="configured architecture bound (25%)")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=35, ha="right")
    ax.set_ylim(0, 27)
    ax.set_ylabel("guard rejection rate (%)")
    ax.set_title("Guarded meta-policy deployment rejection rates")
    ax.legend(loc="upper right")
    save(fig, output)


def deployment_curves_plot(results: dict[str, Any], output: Path) -> None:
    targets = (("width", "2x128"), ("width", "2x512"), ("depth", "8x128"))
    methods = METHODS
    fig, axes = plt.subplots(1, len(targets), figsize=(13.2, 4.2), sharey=False)
    for ax, (axis, spec) in zip(axes, targets):
        selected = [row for row in results["individual_results"]
                    if row["axis"] == axis and row["spec"] == spec]
        for method, name, color, _ in methods:
            curves = [row["loss_curve"] for row in selected if row["method"] == method]
            if not curves:
                continue
            values = np.asarray(curves, dtype=float)
            mean = values.mean(axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, label=name, color=color,
                    linewidth=1.35)
        ax.set_yscale("log")
        ax.set_title(f'{axis} {spec}')
        ax.set_xlabel("deployment update")
        ax.set_ylabel("mean training CE (log scale)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Deployment loss curves averaged over evaluation task seeds (n=3)")
    save(fig, output)


def operations_plot(events: list[dict[str, Any]], output: Path) -> None:
    start = dt.datetime.fromisoformat(events[0]["timestamp_utc"].replace("Z", "+00:00"))
    rows = []
    for event in events:
        gpu = event.get("data", {}).get("gpu")
        if event["event"] != "heartbeat" or not isinstance(gpu, dict):
            continue
        timestamp = dt.datetime.fromisoformat(event["timestamp_utc"].replace("Z", "+00:00"))
        rows.append(((timestamp - start).total_seconds() / 60.0, gpu))
    if not rows:
        raise RuntimeError("event log has no heartbeat GPU telemetry")
    minutes = np.array([row[0] for row in rows])
    allocated = np.array([row[1]["allocated_bytes"] / 2**30 for row in rows])
    reserved = np.array([row[1]["reserved_bytes"] / 2**30 for row in rows])
    utilization = np.array([row[1].get("utilization_percent", np.nan) for row in rows])
    temperature = np.array([row[1].get("temperature_c", np.nan) for row in rows])
    fig, (memory_ax, health_ax) = plt.subplots(2, 1, figsize=(10.4, 6.2), sharex=True)
    memory_ax.plot(minutes, allocated, color="#1f77b4", label="allocated")
    memory_ax.plot(minutes, reserved, color="#ff7f0e", label="reserved")
    memory_ax.set_ylabel("GPU memory (GiB)")
    memory_ax.legend(loc="upper right")
    health_ax.plot(minutes, utilization, color="#2ca02c", label="utilization (%)")
    health_ax.plot(minutes, temperature, color="#d62728", label="temperature (°C)")
    health_ax.set_xlabel("minutes since run creation (includes stopped intervals)")
    health_ax.set_ylabel("percent / °C")
    health_ax.set_ylim(0, 105)
    health_ax.legend(loc="upper right")
    fig.suptitle("Heartbeat GPU telemetry across the production run")
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=Path("build/runs/production-seed1-v2"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("docs/assets/report_07-12-26"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load_json(args.run_dir / "results.json")
    analysis = load_json(args.run_dir / "reanalysis.json")
    with (args.run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    if results["run_id"] != analysis["run_id"]:
        raise RuntimeError("results and reanalysis run IDs do not match")
    for filename, key in (("results.json", "results_sha256"),
                          ("events.jsonl", "events_sha256"),
                          ("termination.json", "termination_sha256")):
        observed = hashlib.sha256((args.run_dir / filename).read_bytes()).hexdigest()
        expected = analysis["raw_artifacts"][key]
        if observed != expected:
            raise RuntimeError(
                f"{filename} does not match reanalysis: {observed} != {expected}")
    configure_style()
    benchmark_plot(analysis, args.output_dir / "test_loss.svg", "loss")
    benchmark_plot(analysis, args.output_dir / "test_accuracy.svg", "accuracy")
    paired_effect_plot(analysis, args.output_dir / "paired_loss_effect.svg")
    development_plot(results, args.output_dir / "development_curve.svg")
    rollback_plot(analysis, args.output_dir / "rollback_horizon.svg")
    guard_plot(analysis, args.output_dir / "guard_rejection.svg")
    deployment_curves_plot(results, args.output_dir / "deployment_curves.svg")
    operations_plot(events, args.output_dir / "gpu_telemetry.svg")
    print(f"wrote 8 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
