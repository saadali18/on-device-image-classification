#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="$PROJECT_ROOT/.venv/bin/python"
LOGS="$PROJECT_ROOT/logs/distillation"
mkdir -p "$LOGS"

# Step 3: architectural scaling at the winning hyperparams (tau* = 8, alpha* = 0.5).
# Run smallest student first so failures surface early. ResNet-20 (w=1) is already
# covered in Step 2a's resnet20_t8_a0p5 cell, so we don't re-run it here.
CONFIGS=(
    "configs/distillation_resnet20_w0p5_t8_a0p5.yaml"   # ~0.07M params (smallest)
    "configs/distillation_resnet32_t8_a0p5.yaml"        # ~0.47M
    "configs/distillation_resnet44_t8_a0p5.yaml"        # ~0.67M
    "configs/distillation_resnet56_t8_a0p5.yaml"        # ~0.86M
    "configs/distillation_resnet20_w2_t8_a0p5.yaml"     # ~1.10M (largest)
)

START_TIME=$(date +%s)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="$LOGS/${name}.log"
    echo
    echo "================================================================"
    echo "STARTING: $name  (log: $logfile)"
    echo "================================================================"
    cell_start=$(date +%s)
    "$PY" scripts/train_distillation.py --config "$cfg" 2>&1 | tee "$logfile"
    cell_end=$(date +%s)
    echo "DONE: $name in $((cell_end - cell_start))s"
done

END_TIME=$(date +%s)
echo
echo "================================================================"
echo "STEP 3 SWEEP COMPLETE: $((END_TIME - START_TIME))s total"
echo "================================================================"
