#!/usr/bin/env python3
"""Regenerates every table in docs/kmnist_snn_vs_cnn.md from the raw CSV logs.

    python3 scripts/depth_report.py docs/data/kmnist

Reads depth_sweep.csv / depth_final.csv (the SNN suite) and torch_sweep.csv /
torch_final.csv (the CNN/MLP suite). A "run" is a (tag, seed) group; per-epoch
rows carry both the online train loss and the post-epoch train-eval /
test-set measurements.
"""
from __future__ import annotations

import collections
import csv
import math
import os
import statistics
import sys

# label -> (family, layer widths) for parameter counts; the torch CSV carries
# its own params column, the SNN one is derived from the architecture.
SNN_ARCH = {
    "snn_d1": [256], "snn_d2": [256] * 2, "snn_d3": [256] * 3, "snn_d4": [256] * 4,
    "snn_w512": [512], "snn_w1024": [1024],
}
ORDER = ["snn_d1", "snn_d2", "snn_d3", "snn_d4", "snn_w512", "snn_w1024",
         "mlp_d1", "mlp_d2", "mlp_d3", "mlp_d4", "mlp_w512", "mlp_w1024",
         "cnn_d1", "cnn_d2", "cnn_d3", "cnn_d4"]
LADDERS = {"SNN": ["snn_d1", "snn_d2", "snn_d3", "snn_d4"],
           "MLP": ["mlp_d1", "mlp_d2", "mlp_d3", "mlp_d4"],
           "CNN": ["cnn_d1", "cnn_d2", "cnn_d3", "cnn_d4"]}
WIDTHS = {"SNN": ["snn_d1", "snn_w512", "snn_w1024"],
          "MLP": ["mlp_d1", "mlp_w512", "mlp_w1024"]}


def snn_params(widths):
    layers = [784] + widths + [10]
    return sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def runs(rows):
    """tag -> seed -> list of epoch rows, epoch-sorted."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        out[r["tag"]][r["seed"]].append(r)
    for tag in out:
        for seed in out[tag]:
            out[tag][seed].sort(key=lambda r: int(r["epoch"]))
    return out


def stat(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, sd


def acc_cell(values):
    m, sd = stat(values)
    return f"{100 * m:.2f}% ±{100 * sd:.2f}"


def collect(by_tag, tag):
    """Per-seed scalars for one configuration."""
    seeds = by_tag.get(tag)
    if not seeds:
        return None
    per = []
    for rows in seeds.values():
        last = rows[-1]
        best = max(rows, key=lambda r: float(r["test_acc"]))
        min_tl = min(rows, key=lambda r: float(r["test_loss"]))
        per.append({
            "final_acc": float(last["test_acc"]),
            "best_acc": float(best["test_acc"]),
            "best_epoch": int(best["epoch"]),
            "min_tl_epoch": int(min_tl["epoch"]),
            "min_tl": float(min_tl["test_loss"]),
            "final_tl": float(last["test_loss"]),
            "final_evl": float(last["train_eval_loss"]),
            "final_eva": float(last["train_eval_acc"]),
            "firing": float(last["firing_rate"]) if "firing_rate" in last else None,
            "sec": statistics.mean(float(r["seconds"]) for r in rows),
        })
    return per


def params_of(tag, torch_rows):
    if tag in SNN_ARCH:
        return snn_params(SNN_ARCH[tag])
    for r in torch_rows:
        if r["tag"] == tag:
            return int(r["params"])
    return None


def winners(path, snn):
    """Re-derive each architecture's swept winner, as the run scripts do."""
    best_epoch = {}
    for r in load(path):
        cfg, lr = r["tag"].removeprefix("sweep_").rsplit("_lr", 1)
        key = (cfg, lr, r.get("alpha", "-"), r["seed"]) if snn else (cfg, lr, "-", r["seed"])
        best_epoch[key] = max(best_epoch.get(key, 0.0), float(r["test_acc"]))
    acc = collections.defaultdict(list)
    for (cfg, lr, alpha, _), v in best_epoch.items():
        acc[(cfg, lr, alpha)].append(v)
    win = {}
    for (cfg, lr, alpha), v in acc.items():
        m = statistics.mean(v)
        if cfg not in win or m > win[cfg][2]:
            win[cfg] = (lr, alpha, m)
    return win


def main_table(by_tag, torch_rows, snn_win, torch_win):
    print("### head to head - KMNIST, 15 epochs, 4 seeds, each at its swept lr (SNN: and alpha)\n")
    print("| model | params | lr | best test acc | final test acc | epoch of min test loss | firing |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for tag in ORDER:
        per = collect(by_tag, tag)
        if per is None:
            continue
        p = params_of(tag, torch_rows)
        fam, _, rest = tag.partition("_")
        wkey = rest if fam == "snn" else None
        if fam == "snn":
            lr, alpha, _ = snn_win.get(wkey, ("?", "?", 0))
            lr = f"{lr}, a={alpha}"
        else:
            arch, kind = fam, rest
            depth = kind[1:] if kind.startswith("d") else "1"
            width = kind[1:] if kind.startswith("w") else ("256" if arch == "mlp" else "0")
            lr = torch_win.get(f"{arch}{depth}w{width}", ("?",))[0]
        firing = per[0]["firing"]
        fire = f"{statistics.mean(x['firing'] for x in per):.3f}" if firing is not None else "--"
        print(f"| `{tag}` | {p:,} | {lr} | **{acc_cell([x['best_acc'] for x in per])}** | "
              f"{acc_cell([x['final_acc'] for x in per])} | "
              f"{statistics.mean(x['min_tl_epoch'] for x in per):.1f} | {fire} |")
    print()


def depth_table(by_tag):
    print("### does depth help? paired against each family's depth-1 net, 4 shared seeds\n")
    print("| family | step | delta best test acc | t (df=3) | p<0.05? |")
    print("| --- | --- | --- | --- | --- |")
    crit = 3.182
    for fam, tags in LADDERS.items():
        base = collect(by_tag, tags[0])
        if base is None:
            continue
        base_acc = [x["best_acc"] for x in base]
        for tag in tags[1:]:
            per = collect(by_tag, tag)
            if per is None:
                continue
            diffs = [b - a for a, b in zip(base_acc, (x["best_acc"] for x in per))]
            m, sd = stat(diffs)
            t = m / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
            print(f"| {fam} | {tags[0][-2:]} -> {tag[len(fam) + 1:]} | {100 * m:+.2f}% | "
                  f"{t:+.2f} | {'yes' if abs(t) > crit else 'no'} |")
    print()


def width_table(by_tag):
    print("### parameters via width instead - depth-1 nets, 4 seeds\n")
    print("| family | width | params | best test acc |")
    print("| --- | --- | --- | --- |")
    for fam, tags in WIDTHS.items():
        for tag, w in zip(tags, ["256", "512", "1024"]):
            per = collect(by_tag, tag)
            if per is None:
                continue
            # the MLP mirrors the SNN's dense shapes, so the count is shared
            print(f"| {fam} | {w} | {snn_params([int(w)]):,} | {acc_cell([x['best_acc'] for x in per])} |")
    print()


def gap_table(by_tag):
    print("### the generalization gap - losses on the final-epoch frozen model, 4 seeds\n")
    print("| model | train loss (10k eval) | test loss | gap ratio | train acc | test acc |")
    print("| --- | --- | --- | --- | --- | --- |")
    for tag in ORDER:
        per = collect(by_tag, tag)
        if per is None:
            continue
        evl = statistics.mean(x["final_evl"] for x in per)
        tl = statistics.mean(x["final_tl"] for x in per)
        eva = statistics.mean(x["final_eva"] for x in per)
        fa = statistics.mean(x["final_acc"] for x in per)
        print(f"| `{tag}` | {evl:.4f} | {tl:.4f} | {tl / evl:.1f}x | "
              f"{100 * eva:.2f}% | {100 * fa:.2f}% |")
    print()


def curve_table(by_tag, tags, metric, title):
    have = [t for t in tags if t in by_tag]
    if not have:
        return
    print(f"### {title}\n")
    epochs = sorted({int(r["epoch"]) for t in have for rows in by_tag[t].values() for r in rows})
    print("| model | " + " | ".join(str(e) for e in epochs) + " |")
    print("| --- |" + " --- |" * len(epochs))
    for t in have:
        cells = []
        for e in epochs:
            vals = [float(r[metric]) for rows in by_tag[t].values() for r in rows if int(r["epoch"]) == e]
            cells.append(f"{statistics.mean(vals):.3f}" if vals else "--")
        print(f"| `{t}` | " + " | ".join(cells) + " |")
    print()


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/data/kmnist"
    snn_rows = load(os.path.join(out, "depth_final.csv"))
    torch_rows = load(os.path.join(out, "torch_final.csv"))
    if not snn_rows and not torch_rows:
        print("no rows loaded", file=sys.stderr)
        return 1
    by_tag = runs(snn_rows + torch_rows)
    snn_win = winners(os.path.join(out, "depth_sweep.csv"), snn=True)
    torch_win = winners(os.path.join(out, "torch_sweep.csv"), snn=False)

    main_table(by_tag, torch_rows, snn_win, torch_win)
    depth_table(by_tag)
    width_table(by_tag)
    gap_table(by_tag)
    curve_table(by_tag, ["snn_d1", "snn_d2", "cnn_d1", "cnn_d2", "mlp_d1"],
                "test_loss", "test loss by epoch (mean over seeds)")
    curve_table(by_tag, ["snn_d1", "snn_d2", "cnn_d1", "cnn_d2", "mlp_d1"],
                "train_eval_loss", "train loss (10k eval) by epoch (mean over seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
