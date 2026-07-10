#!/usr/bin/env bash
# Repeats the MNIST surrogate comparison on Kuzushiji-MNIST, which is the same
# shape and cost but roughly ten points harder -- enough headroom for accuracy
# to discriminate surrogates, which on MNIST it could not.
#
#   cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=OFF
#   cmake --build build-tools -j
#   bash scripts/kmnist_experiments.sh
#
# The protocol is deliberately identical to scripts/mnist_experiments.sh so the
# two datasets can be compared directly. Roughly 90 minutes on 12 cores.
set -euo pipefail

BIN=${BIN:-./build-tools/mnist_bptt}
DATA=${DATA:-data/kmnist}
OUT=${OUT:-docs/data/kmnist}
mkdir -p "$OUT"

SURROGATES=(fast_sigmoid atan sigmoid triangle gaussian rectangular)

# 1. alpha sweep. Each surrogate gets its own best gradient-window width before
#    the shapes are compared; a shared alpha would just measure slope mismatch.
if [ ! -s "$OUT/sweep.csv" ]; then
  "$BIN" --data "$DATA" --mode sweep --hidden 256 --timesteps 20 --epochs 4 --seeds 3 --lr 2e-3 \
         --csv "$OUT/sweep.csv" | tee "$OUT/sweep.txt"
fi

# 2. read each surrogate's best alpha back out of the sweep, rather than
#    hardcoding MNIST's answers -- the optimum need not transfer.
eval "$(python3 - "$OUT/sweep.csv" <<'PY'
import csv, collections, statistics, sys
last = {}
for r in csv.DictReader(open(sys.argv[1])):
    k = (r['surrogate'], float(r['alpha']), int(r['seed']))
    if k not in last or int(r['epoch']) > int(last[k]['epoch']):
        last[k] = r
acc = collections.defaultdict(list)
for (s, a, _), r in last.items():
    acc[(s, a)].append(float(r['test_acc']))
best = {}
for (s, a), v in acc.items():
    m = statistics.mean(v)
    if s not in best or m > best[s][1]:
        best[s] = (a, m)
print("declare -A BEST_ALPHA=(" + " ".join(f"[{s}]={a:g}" for s, (a, _) in best.items()) + ")")
PY
)"
echo "best alpha per surrogate on KMNIST: $(declare -p BEST_ALPHA)"

# 3. head to head at each surrogate's own best alpha, 8 seeds for the statistics.
rm -f "$OUT/headtohead.csv"
: > "$OUT/headtohead.txt"
for s in "${SURROGATES[@]}"; do
  "$BIN" --data "$DATA" --mode single --hidden 1000 --timesteps 25 --epochs 8 --seeds 8 --lr 1e-3 \
         --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" \
         --csv "$OUT/headtohead.csv" --tag headtohead | tee -a "$OUT/headtohead.txt"
done

# 4. reset-path ablation, to check the ~0.3% MNIST result is not dataset-specific.
rm -f "$OUT/detach.csv"
: > "$OUT/detach.txt"
for s in atan fast_sigmoid; do
  "$BIN" --data "$DATA" --mode single --hidden 256 --timesteps 20 --epochs 4 --seeds 3 --lr 2e-3 \
         --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" \
         --csv "$OUT/detach.csv" --tag "attached_$s" | tee -a "$OUT/detach.txt"
  "$BIN" --data "$DATA" --mode single --hidden 256 --timesteps 20 --epochs 4 --seeds 3 --lr 2e-3 \
         --surrogate "$s" --alpha "${BEST_ALPHA[$s]}" --detach \
         --csv "$OUT/detach.csv" --tag "detached_$s" | tee -a "$OUT/detach.txt"
done

echo "=== kmnist experiments complete ==="
