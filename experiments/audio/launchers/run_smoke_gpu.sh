#!/usr/bin/env bash
# GPU smoke rungs (P0): armf-tiny -> overfit -> depth0, sequentially.
set -euo pipefail
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
cd /nlp/scr/potsawee/workspace/soda-extension/marin
for rung in armf-tiny overfit depth0; do
  echo "=== RUNG $rung ==="
  .venv/bin/python experiments/audio/exp_smoke.py --rung $rung
done
echo "ALL GPU SMOKE RUNGS PASS"
