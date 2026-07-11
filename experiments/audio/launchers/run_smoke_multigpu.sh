#!/usr/bin/env bash
# Multi-GPU plumbing test: run both arms' tiny smokes across the visible GPUs.
set -euo pipefail
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0 XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
cd /nlp/scr/potsawee/workspace/soda-extension/marin
echo "visible GPUs:"; .venv/bin/python -c "import jax; print(jax.devices())"
for rung in armf-tiny overfit; do
  echo "=== $rung (multi-GPU) ==="
  .venv/bin/python experiments/audio/exp_smoke.py --rung $rung
done
echo "MULTI-GPU SMOKE PASS"
