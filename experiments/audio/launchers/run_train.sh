#!/usr/bin/env bash
# Launch one isoflop training run: run_train.sh <run-name-from-exp-script> [exp-script]
# e.g.  run_train.sh p1-hier            (exp_isoflop_headline.py)
set -euo pipefail
RUN="$1"
SCRIPT="${2:-exp_isoflop_headline.py}"
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=${LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS:-1}
# The fused-CE batched_xla path peaks at ~18 GiB of temporaries for the largest
# flat config (B=16 x 4096 x 144,644 vocab). JAX's default 0.75 pool (36 GiB of
# 48) OOMed p2-flat-d512 at step 10; 0.92 leaves ~4 GiB for CUDA context.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
cd /nlp/scr/potsawee/workspace/soda-extension/marin
exec .venv/bin/python "experiments/audio/$SCRIPT" --run "$RUN"
