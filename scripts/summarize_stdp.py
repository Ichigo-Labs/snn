#!/usr/bin/env python3
"""Regenerate the tables in docs/kmnist_stdp.md from a stdp_final.csv.

Reads the per-seed CSV written by tools/kmnist_stdp.py and prints the readout
table (per layer + concat + raw-pixel control, untrained vs trained), the paired
per-seed significance contrasts, and the per-layer physical state. Pure stdlib.

Usage:
    python3 scripts/summarize_stdp.py docs/data/kmnist/stdp_final.csv
"""

import csv
import math
import sys

HIDDEN_LAYERS = 4
# Two-sided Student t critical values at p=0.05, indexed by degrees of freedom.
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def mean_sd(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def paired(after, before):
    """Mean paired difference and its Student t over matched seeds."""
    deltas = [a - b for a, b in zip(after, before)]
    mean, sd = mean_sd(deltas)
    t = mean / (sd / math.sqrt(len(deltas))) if sd > 0 else math.inf
    return mean, t


def load(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"summarize_stdp: {path} has no data rows")
    return rows


def column(rows, name, scale=100.0):
    return [float(row[name]) * scale for row in rows]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_stdp.py <stdp_final.csv>")
    rows = load(sys.argv[1])
    seeds = len(rows)
    df = seeds - 1
    crit = T_CRITICAL.get(df)
    print(f"# {sys.argv[1]}  ({seeds} seeds, {rows[0]['epochs']} epochs, "
          f"df={df}, crit t={crit})\n")

    pixel = column(rows, "pixel_test_acc")
    pixel_mean, _ = mean_sd(pixel)
    untrained = {i: column(rows, f"baseline_layer{i}_test_acc") for i in range(1, HIDDEN_LAYERS + 1)}
    trained = {i: column(rows, f"final_layer{i}_test_acc") for i in range(1, HIDDEN_LAYERS + 1)}
    train_acc = {i: column(rows, f"final_layer{i}_train_acc") for i in range(1, HIDDEN_LAYERS + 1)}
    concat_u = column(rows, "baseline_concat_test_acc")
    concat_t = column(rows, "final_concat_test_acc")

    print("## Readout test accuracy (mean +/- sd)")
    print(f"  {'raw pixels':16s} {pixel_mean:6.2f}")
    for i in range(1, HIDDEN_LAYERS + 1):
        um, _ = mean_sd(untrained[i])
        tm, ts = mean_sd(trained[i])
        print(f"  {'hidden L' + str(i):16s} untrained {um:6.2f}   trained {tm:6.2f} +/- {ts:4.2f}"
              f"   (train {mean_sd(train_acc[i])[0]:6.2f})")
    cum, _ = mean_sd(concat_u)
    ctm, cts = mean_sd(concat_t)
    print(f"  {'concat L1-L4':16s} untrained {cum:6.2f}   trained {ctm:6.2f} +/- {cts:4.2f}")

    print("\n## Paired per-seed contrasts")
    contrasts = [
        ("STDP learning: trained - untrained, L1", trained[1], untrained[1]),
        ("STDP L1 - raw pixels", trained[1], pixel),
        ("STDP concat - raw pixels", concat_t, pixel),
        ("untrained concat - raw pixels", concat_u, pixel),
        ("depth cost: trained L1 - L4", trained[1], trained[HIDDEN_LAYERS]),
    ]
    for label, after, before in contrasts:
        mean, t = paired(after, before)
        verdict = "significant" if crit and abs(t) > crit else "n.s."
        print(f"  {label:40s} {mean:+6.2f}%   t={t:+6.1f}   {verdict}")

    print("\n## Per-layer physical state (trained, mean over seeds)")
    print(f"  {'layer':6s} {'fire':>7s} {'dead%':>7s} {'|dW|/|W0|%':>11s} {'theta':>7s}")
    for i in range(1, HIDDEN_LAYERS + 1):
        fire, _ = mean_sd([float(row[f"layer{i}_train_rate"]) for row in rows])
        dead, _ = mean_sd(column(rows, f"layer{i}_dead_fraction"))
        delta, _ = mean_sd(column(rows, f"layer{i}_weight_delta"))
        theta, _ = mean_sd([float(row[f"layer{i}_threshold"]) for row in rows])
        print(f"  L{i:<5d} {fire:7.4f} {dead:7.2f} {delta:11.2f} {theta:7.3f}")


if __name__ == "__main__":
    main()
