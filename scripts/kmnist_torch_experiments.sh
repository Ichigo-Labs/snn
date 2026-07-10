#!/usr/bin/env bash
# Conventional-network side of the depth benchmark: a plain CNN ladder and a
# ReLU-MLP control (widths matched to the SNN's hidden layers) on KMNIST,
# trained by tools/kmnist_cnn.py under the same protocol as
# scripts/kmnist_depth_experiments.sh — lr sweep first, then 15-epoch finals
# on 4 shared seeds with the post-epoch train-eval measurement.
#
# Minutes on a GPU; the script deliberately leaves the CPU cores free so the
# SNN suite can run at the same time.
set -euo pipefail

PY=${PY:-python3}
TOOL=${TOOL:-tools/kmnist_cnn.py}
DATA=${DATA:-data/kmnist}
OUT=${OUT:-docs/data/kmnist}
mkdir -p "$OUT"

# cnn depth = conv3x3-ReLU-pool blocks; mlp mirrors the SNN ladder exactly.
CONFIGS=(
  "cnn 1 0" "cnn 2 0" "cnn 3 0" "cnn 4 0"
  "mlp 1 256" "mlp 2 256" "mlp 3 256" "mlp 4 256"
  "mlp 1 512" "mlp 1 1024"
)

LRS=(3e-4 1e-3 3e-3)

# 1. lr sweep, 4 epochs x 2 seeds, matching the SNN sweep's budget.
if [ ! -s "$OUT/torch_sweep.csv" ]; then
  for cfg in "${CONFIGS[@]}"; do
    read -r arch depth width <<<"$cfg"
    for lr in "${LRS[@]}"; do
      "$PY" "$TOOL" --data "$DATA" --arch "$arch" --depth "$depth" --width "$width" \
            --epochs 4 --seeds 2 --lr "$lr" \
            --csv "$OUT/torch_sweep.csv" --tag "sweep_${arch}${depth}w${width}_lr${lr}"
    done
  done
fi

# 2. pick each configuration's lr by mean best-epoch test accuracy.
eval "$(python3 - "$OUT/torch_sweep.csv" <<'PY'
import csv, collections, statistics, sys
best_epoch = {}
for r in csv.DictReader(open(sys.argv[1])):
    cfg, lr = r["tag"].removeprefix("sweep_").rsplit("_lr", 1)
    k = (cfg, lr, r["seed"])
    best_epoch[k] = max(best_epoch.get(k, 0.0), float(r["test_acc"]))
acc = collections.defaultdict(list)
for (cfg, lr, _), v in best_epoch.items():
    acc[(cfg, lr)].append(v)
win = {}
for (cfg, lr), v in acc.items():
    m = statistics.mean(v)
    if cfg not in win or m > win[cfg][1]:
        win[cfg] = (lr, m)
print("declare -A BEST_LR=(" + " ".join(f"[{c}]={lr}" for c, (lr, _) in win.items()) + ")")
PY
)"
echo "winners: $(declare -p BEST_LR)"

# 3. finals: identical protocol to the SNN finals.
rm -f "$OUT/torch_final.csv"
: > "$OUT/torch_final.txt"
for cfg in "${CONFIGS[@]}"; do
  read -r arch depth width <<<"$cfg"
  key="${arch}${depth}w${width}"
  tag="${arch}_d${depth}"
  if [ "$arch" = mlp ] && [ "$width" != 256 ]; then
    tag="mlp_w${width}"
  fi
  "$PY" "$TOOL" --data "$DATA" --arch "$arch" --depth "$depth" --width "$width" \
        --epochs 15 --seeds 4 --lr "${BEST_LR[$key]}" --train-eval 10000 \
        --csv "$OUT/torch_final.csv" --tag "$tag" | tee -a "$OUT/torch_final.txt"
done

echo "=== kmnist torch experiments complete ==="
