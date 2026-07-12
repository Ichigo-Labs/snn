#!/usr/bin/env bash
# Meta-learned local plasticity rule vs STDP on KMNIST. A >=50M-param policy is
# trained by gradient imitation to emit local weight updates; it is benchmarked
# against STDP and the surrogate-GD teacher, and against synthetic directions of
# controlled cosine to the true gradient. CUDA required (the tool refuses CPU).
set -euo pipefail

PYTHON=${PYTHON:-python3}
DATA=${DATA:-data/kmnist}
OUT=${OUT:-docs/data/kmnist}
SEEDS=${SEEDS:-2}
META_STEPS=${META_STEPS:-5000}

mkdir -p "$OUT"
rm -f "$OUT"/metaplasticity_final.csv "$OUT"/metaplasticity_final.txt \
      "$OUT"/metaplasticity_cosine_curve.txt "$OUT"/metaplasticity_self_test.txt

"$PYTHON" tools/kmnist_metaplasticity.py --self-test | tee "$OUT/metaplasticity_self_test.txt"

# The control that interprets everything: how much gradient-cosine does a
# deployment actually need? Synthetic directions, fresh (unbiased) noise per step.
"$PYTHON" tools/kmnist_metaplasticity.py --data "$DATA" \
  --cosine-curve "1.0,0.6,0.4,0.3,0.2,0.1,0.0" --deploy-epochs 3 --seed0 1 \
  | tee "$OUT/metaplasticity_cosine_curve.txt"

# The learned three-factor rule vs the surrogate-GD teacher, over SEEDS deploys.
"$PYTHON" tools/kmnist_metaplasticity.py --data "$DATA" --factor three \
  --meta-steps "$META_STEPS" --meta-log-every 500 --deploy-epochs 3 \
  --seeds "$SEEDS" --seed0 1 --include-teacher \
  --csv "$OUT/metaplasticity_final.csv" --tag meta3 \
  | tee "$OUT/metaplasticity_final.txt"
