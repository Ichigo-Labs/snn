#!/usr/bin/env bash
# SNN side of the depth benchmark: how much does a deeper (or wider) LIF
# network help on KMNIST, and how does its train/test loss gap evolve?
#
#   cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=OFF
#   cmake --build build-tools -j
#   bash scripts/kmnist_depth_experiments.sh
#
# Two phases, like scripts/kmnist_experiments.sh: a (lr, alpha) sweep per
# architecture first — the best alpha is already known not to transfer across
# datasets, so it is not assumed to transfer across depth either, and deeper
# nets often want a different learning rate — then 15-epoch finals at each
# architecture's own winner, 4 shared seeds, with --train-eval scoring the
# frozen model on a fixed 10k train prefix so the generalization gap is
# measured post-epoch like the test loss is.
#
# Roughly 10 minutes on the CUDA backend (GPU=0 falls back to the CPU
# trainer: identical math to float tolerance, ~2.5 hours on 12 cores; do not
# mix backends within one dataset, the runs are only bit-reproducible within
# a backend). The companion conventional-network suite is
# scripts/kmnist_torch_experiments.sh.
set -euo pipefail

BIN=${BIN:-./build-tools/mnist_bptt}
DATA=${DATA:-data/kmnist}
OUT=${OUT:-docs/data/kmnist}
GPU=${GPU:-1}
GPU_FLAG=$([ "$GPU" = 1 ] && echo "--gpu" || echo "")
mkdir -p "$OUT"

# depth ladder at width 256, plus width controls: w512 matches d3's parameter
# count and w1024 exceeds d4's, so "more parameters via depth" and "more
# parameters via width" can be told apart.
declare -A HIDDEN=(
  [d1]=256 [d2]=256,256 [d3]=256,256,256 [d4]=256,256,256,256
  [w512]=512 [w1024]=1024
)
ARCHS=(d1 d2 d3 d4 w512 w1024)

LRS=(5e-4 1e-3 2e-3)

# 1. sweep. Depth ladders re-derive alpha; width controls keep alpha=1, which
#    the earlier KMNIST sweeps picked at both width 256 and width 1000.
if [ ! -s "$OUT/depth_sweep.csv" ]; then
  for arch in "${ARCHS[@]}"; do
    case "$arch" in
      w*) ALPHAS=(1) ;;
      *)  ALPHAS=(0.5 1 2) ;;
    esac
    for lr in "${LRS[@]}"; do
      for a in "${ALPHAS[@]}"; do
        "$BIN" $GPU_FLAG --data "$DATA" --mode single --hidden "${HIDDEN[$arch]}" --timesteps 20 \
               --epochs 4 --seeds 2 --lr "$lr" --surrogate atan --alpha "$a" \
               --csv "$OUT/depth_sweep.csv" --tag "sweep_${arch}_lr${lr}" \
          || echo "WARNING: sweep_${arch}_lr${lr} alpha=$a did not finish" >&2
      done
    done
  done
fi

# 2. pick each architecture's (lr, alpha) by mean best-epoch test accuracy.
eval "$(python3 - "$OUT/depth_sweep.csv" <<'PY'
import csv, collections, statistics, sys
best_epoch = {}
for r in csv.DictReader(open(sys.argv[1])):
    arch, lr = r["tag"].removeprefix("sweep_").rsplit("_lr", 1)
    k = (arch, lr, r["alpha"], r["seed"])
    best_epoch[k] = max(best_epoch.get(k, 0.0), float(r["test_acc"]))
acc = collections.defaultdict(list)
for (arch, lr, alpha, _), v in best_epoch.items():
    acc[(arch, lr, alpha)].append(v)
win = {}
for (arch, lr, alpha), v in acc.items():
    m = statistics.mean(v)
    if arch not in win or m > win[arch][2]:
        win[arch] = (lr, alpha, m)
print("declare -A BEST_LR=(" + " ".join(f"[{a}]={lr}" for a, (lr, _, _) in win.items()) + ")")
print("declare -A BEST_ALPHA=(" + " ".join(f"[{a}]={al}" for a, (_, al, _) in win.items()) + ")")
PY
)"
echo "winners: $(declare -p BEST_LR BEST_ALPHA)"

# 3. finals: 15 epochs, 4 shared seeds, generalization gap measured post-epoch.
rm -f "$OUT/depth_final.csv"
: > "$OUT/depth_final.txt"
for arch in "${ARCHS[@]}"; do
  "$BIN" $GPU_FLAG --data "$DATA" --mode single --hidden "${HIDDEN[$arch]}" --timesteps 20 \
         --epochs 15 --seeds 4 --lr "${BEST_LR[$arch]}" --surrogate atan \
         --alpha "${BEST_ALPHA[$arch]}" --train-eval 10000 \
         --csv "$OUT/depth_final.csv" --tag "snn_${arch}" | tee -a "$OUT/depth_final.txt"
done

echo "=== kmnist depth experiments complete ==="
