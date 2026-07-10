#!/usr/bin/env python3
"""Regenerates every table in docs/mnist_bptt.md from the raw CSV logs.

    python3 scripts/mnist_report.py docs/data/*.csv

Each CSV row is one epoch of one run:
    tag,surrogate,alpha,seed,epoch,train_loss,test_loss,test_acc,firing_rate,seconds
A "run" is a (tag, surrogate, alpha, seed) group; its result is its last epoch.
"""
from __future__ import annotations

import collections
import csv
import math
import statistics
import sys

SURROGATES = ["fast_sigmoid", "atan", "sigmoid", "triangle", "gaussian", "rectangular"]
FLOATS = ("alpha", "train_loss", "test_loss", "test_acc", "firing_rate", "seconds")
INTS = ("seed", "epoch")


def load(paths):
    rows = []
    for path in paths:
        try:
            fh = open(path, newline="")
        except OSError:
            continue
        with fh:
            for row in csv.DictReader(fh):
                for k in FLOATS:
                    row[k] = float(row[k])
                for k in INTS:
                    row[k] = int(row[k])
                rows.append(row)
    return rows


def runs(rows, tag_filter=None):
    """Collapse each run to its final epoch."""
    last = {}
    for r in rows:
        if tag_filter and not tag_filter(r["tag"]):
            continue
        key = (r["tag"], r["surrogate"], r["alpha"], r["seed"])
        if key not in last or r["epoch"] > last[key]["epoch"]:
            last[key] = r
    return list(last.values())


def stat(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def cell(values, scale=100.0, unit="%"):
    m, sd = stat(values)
    return f"{scale * m:.2f}{unit} ±{scale * sd:.2f}"


def sweep_table(rows):
    group = collections.defaultdict(list)
    for r in runs(rows, lambda t: t == "sweep"):
        group[(r["surrogate"], r["alpha"])].append(r["test_acc"])
    if not group:
        return
    alphas = sorted({a for _, a in group})
    print("### alpha sweep - 784-256-10, T=20, 4 epochs, 3 seeds, lr=2e-3\n")
    print("| surrogate | " + " | ".join(f"a={a:g}" for a in alphas) + " | best a | spread |")
    print("| --- |" + " --- |" * (len(alphas) + 2))
    for s in SURROGATES:
        cells, means = [], {}
        for a in alphas:
            vals = group.get((s, a))
            if not vals:
                cells.append("--")
                continue
            m, sd = stat(vals)
            means[a] = m
            cells.append(f"{100 * m:.2f} ±{100 * sd:.2f}")
        if not means:
            continue
        best = max(means, key=means.get)
        spread = 100 * (max(means.values()) - min(means.values()))
        print(f"| `{s}` | " + " | ".join(cells) + f" | **{best:g}** | {spread:.2f} |")
    print()
    fire = collections.defaultdict(list)
    for r in runs(rows, lambda t: t == "sweep"):
        fire[r["alpha"]].append(r["firing_rate"])
    print("Hidden firing rate, averaged over surrogates and seeds: "
          + ", ".join(f"a={a:g} -> {statistics.mean(v):.3f}" for a, v in sorted(fire.items())) + "\n")


def headtohead_table(rows):
    sel = runs(rows, lambda t: t == "headtohead")
    if not sel:
        return
    group = collections.defaultdict(list)
    for r in sel:
        group[r["surrogate"]].append(r)
    seeds = sorted({r["seed"] for r in sel})
    print(f"### head to head - 784-1000-10, T=25, 8 epochs, {len(seeds)} seeds, lr=1e-3, each at its own best alpha\n")
    # Wall-clock per epoch is deliberately not reported per surrogate: repeated
    # measurements on this machine reorder the six, so the differences sit below
    # run-to-run drift.
    print("| surrogate | alpha | final test accuracy | best single run | firing rate |")
    print("| --- | --- | --- | --- | --- |")
    ranked = sorted(group.items(), key=lambda kv: -statistics.mean(r["test_acc"] for r in kv[1]))
    for s, rs in ranked:
        accs = [r["test_acc"] for r in rs]
        print(f"| `{s}` | {rs[0]['alpha']:g} | **{cell(accs)}** | {100 * max(accs):.2f}% | "
              f"{statistics.mean(r['firing_rate'] for r in rs):.3f} |")
    print()

    # Paired comparison against the leader: the seeds are shared across surrogates.
    leader = ranked[0][0]
    by_seed = {(r["surrogate"], r["seed"]): r["test_acc"] for r in sel}
    print(f"Paired difference against `{leader}` over the {len(seeds)} shared seeds "
          f"(Student t, two-sided, df={len(seeds) - 1}):\n")
    print("| surrogate | mean gap | sd | t | p<0.05? |")
    print("| --- | --- | --- | --- | --- |")
    crit = {2: 4.303, 4: 2.776, 7: 2.365, 9: 2.262}.get(len(seeds) - 1, 2.365)
    for s, _ in ranked[1:]:
        diffs = [by_seed[(leader, sd)] - by_seed[(s, sd)] for sd in seeds if (s, sd) in by_seed]
        if len(diffs) < 2:
            continue
        m, sd = stat(diffs)
        t = m / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
        print(f"| `{s}` | {100 * m:+.3f}% | {100 * sd:.3f} | {t:+.2f} | {'yes' if abs(t) > crit else 'no'} |")
    print()

    # Convergence: mean test accuracy after each epoch.
    curve = collections.defaultdict(list)
    for r in rows:
        if r["tag"] == "headtohead":
            curve[(r["surrogate"], r["epoch"])].append(r["test_acc"])
    epochs = sorted({e for _, e in curve})
    print("Test accuracy after each epoch (mean over seeds):\n")
    print("| surrogate | " + " | ".join(f"{e}" for e in epochs) + " |")
    print("| --- |" + " --- |" * len(epochs))
    for s in SURROGATES:
        cells = [f"{100 * statistics.mean(curve[(s, e)]):.2f}" if (s, e) in curve else "--" for e in epochs]
        print(f"| `{s}` | " + " | ".join(cells) + " |")
    print()


def timestep_table(rows):
    sel = runs(rows, lambda t: t.startswith("T") and t[1:].isdigit())
    if not sel:
        return
    group = collections.defaultdict(list)
    for r in sel:
        group[int(r["tag"][1:])].append(r)
    print("### unrolled depth - 784-256-10, gaussian a=1, 4 epochs, 2 seeds\n")
    print("| timesteps | test accuracy | firing rate | s/epoch |")
    print("| --- | --- | --- | --- |")
    for T in sorted(group):
        rs = group[T]
        print(f"| {T} | {cell([r['test_acc'] for r in rs])} | "
              f"{statistics.mean(r['firing_rate'] for r in rs):.3f} | "
              f"{statistics.mean(r['seconds'] for r in rs):.2f} |")
    print()


def detach_table(rows):
    sel = runs(rows, lambda t: t.startswith(("attached_", "detached_")))
    if not sel:
        return
    group = collections.defaultdict(list)
    for r in sel:
        kind, _, surrogate = r["tag"].partition("_")
        group[(surrogate, kind)].append(r)
    print("### the reset path - 784-256-10, T=20, 4 epochs, 3 seeds\n")
    print("| surrogate | reset gradient | test accuracy | firing rate |")
    print("| --- | --- | --- | --- |")
    for surrogate in sorted({s for s, _ in group}):
        for kind in ("attached", "detached"):
            rs = group.get((surrogate, kind))
            if rs:
                label = "backpropagated" if kind == "attached" else "detached"
                print(f"| `{surrogate}` | {label} | {cell([r['test_acc'] for r in rs])} | "
                      f"{statistics.mean(r['firing_rate'] for r in rs):.3f} |")
    print()


def lr_table(rows):
    sel = runs(rows, lambda t: t.startswith("lr"))
    if not sel:
        return
    group = collections.defaultdict(list)
    for r in sel:
        lr, _, surrogate = r["tag"][2:].partition("_")
        group[(surrogate, float(lr))].append(r["test_acc"])
    lrs = sorted({lr for _, lr in group})
    print("### learning-rate sensitivity - 784-256-10, T=20, 3 epochs, 2 seeds\n")
    print("| surrogate | " + " | ".join(f"lr={lr:g}" for lr in lrs) + " |")
    print("| --- |" + " --- |" * len(lrs))
    for surrogate in sorted({s for s, _ in group}):
        cells = [cell(group[(surrogate, lr)]) if (surrogate, lr) in group else "--" for lr in lrs]
        print(f"| `{surrogate}` | " + " | ".join(cells) + " |")
    print()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: mnist_report.py <csv>...", file=sys.stderr)
        return 2
    rows = load(sys.argv[1:])
    if not rows:
        print("no rows loaded", file=sys.stderr)
        return 1
    sweep_table(rows)
    headtohead_table(rows)
    timestep_table(rows)
    detach_table(rows)
    lr_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
