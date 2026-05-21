#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="$PROJECT_ROOT/.venv/bin/python"
LOGS="$PROJECT_ROOT/logs/distillation"
mkdir -p "$LOGS"

CONFIGS=(
    "configs/distillation_resnet20_t2_a0p5.yaml"
    "configs/distillation_resnet20_t4_a0p5.yaml"
    "configs/distillation_resnet20_t8_a0p5.yaml"
    "configs/distillation_resnet20_t16_a0p5.yaml"
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
echo "STEP 2A SWEEP COMPLETE: $((END_TIME - START_TIME))s total"
echo "================================================================"
