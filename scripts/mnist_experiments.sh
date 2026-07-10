#!/usr/bin/env bash
# Reproduces every number in docs/mnist_bptt.md.
#
#   cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=OFF
#   cmake --build build-tools -j
#   bash scripts/mnist_experiments.sh
#
# Writes per-epoch records to docs/data/*.csv and the console summaries to
# docs/data/*.txt. Runtime is roughly 45 minutes on 12 cores.
set -euo pipefail

BIN=${BIN:-./build-tools/mnist_bptt}
OUT=${OUT:-docs/data}
mkdir -p "$OUT"

# Each surrogate's own best gradient-window width, from the alpha sweep below.
# Comparing shapes at one shared alpha would just measure slope mismatch.
declare -A BEST_ALPHA=(
  [fast_sigmoid]=5 [atan]=2 [sigmoid]=2 [triangle]=1 [gaussian]=1 [rectangular]=0.5
)
SURROGATES=(fast_sigmoid atan sigmoid triangle gaussian rectangular)

# 1. alpha sweep: every surrogate x every alpha, 784-256-10.
if [ ! -s "$OUT/sweep.csv" ]; then
  "$BIN" --mode sweep --hidden 256 --timesteps 20 --epochs 4 --seeds 3 --lr 2e-3 \
         --csv "$OUT/sweep.csv" | tee "$OUT/sweep.txt"
fi

# 2. head to head: each surrogate at its own best alpha, 784-1000-10, full data.
rm -f "$OUT/headtohead.csv"
: > "$OUT/headtohead.txt"
for s in "${SURROGATES[@]}"; do
  "$BIN" --mode single --hidden 1000 --timesteps 25 --epochs 8 --seeds 3 --lr 1e-3 \
         --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" \
         --csv "$OUT/headtohead.csv" --tag headtohead | tee -a "$OUT/headtohead.txt"
done

# 3. timestep ablation: how much does unrolling depth buy?
rm -f "$OUT/timesteps.csv"
: > "$OUT/timesteps.txt"
for t in 1 2 5 10 20 30; do
  "$BIN" --mode single --hidden 256 --timesteps "$t" --epochs 4 --seeds 2 --lr 2e-3 \
         --surrogate gaussian --alpha 1 \
         --csv "$OUT/timesteps.csv" --tag "T$t" | tee -a "$OUT/timesteps.txt"
done

# 4. reset-path ablation: does backpropagating through the spike's own reset matter?
rm -f "$OUT/detach.csv"
: > "$OUT/detach.txt"
for s in fast_sigmoid gaussian; do
  "$BIN" --mode single --hidden 256 --timesteps 20 --epochs 4 --seeds 3 --lr 2e-3 \
         --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" \
         --csv "$OUT/detach.csv" --tag "attached_$s" | tee -a "$OUT/detach.txt"
  "$BIN" --mode single --hidden 256 --timesteps 20 --epochs 4 --seeds 3 --lr 2e-3 \
         --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" --detach \
         --csv "$OUT/detach.csv" --tag "detached_$s" | tee -a "$OUT/detach.txt"
done

# 5. learning-rate sensitivity: is the surrogate ranking an artifact of one lr?
rm -f "$OUT/lr.csv"
: > "$OUT/lr.txt"
for s in gaussian rectangular; do
  for lr in 5e-4 1e-3 2e-3 4e-3; do
    "$BIN" --mode single --hidden 256 --timesteps 20 --epochs 3 --seeds 2 --lr "$lr" \
           --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" \
           --csv "$OUT/lr.csv" --tag "lr${lr}_$s" | tee -a "$OUT/lr.txt"
  done
done

echo "=== all experiments complete ==="
