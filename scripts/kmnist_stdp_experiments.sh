#!/usr/bin/env bash
# Four-hidden-layer local-STDP benchmark on the full KMNIST train/test sets.
# PyTorch with a working CUDA device is required; the tool refuses CPU runs.
set -euo pipefail

PYTHON=${PYTHON:-python3}
DATA=${DATA:-data/kmnist}
OUT=${OUT:-docs/data/kmnist}
SEEDS=${SEEDS:-4}
EPOCHS=${EPOCHS:-3}

mkdir -p "$OUT"
rm -f "$OUT/stdp_final.csv" "$OUT/stdp_final.txt"

"$PYTHON" tools/kmnist_stdp.py --self-test | tee "$OUT/stdp_self_test.txt"
"$PYTHON" tools/kmnist_stdp.py --data "$DATA" --epochs "$EPOCHS" \
  --seeds "$SEEDS" --seed0 1 --csv "$OUT/stdp_final.csv" --tag stdp_d4 \
  | tee "$OUT/stdp_final.txt"
