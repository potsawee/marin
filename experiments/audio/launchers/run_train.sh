#!/usr/bin/env bash
# Launch one isoflop training run: run_train.sh <run-name-from-exp-script> [exp-script]
# e.g.  run_train.sh p1-hier            (exp_isoflop_headline.py)
set -euo pipefail
RUN="$1"
SCRIPT="${2:-exp_isoflop_headline.py}"
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=${LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS:-1}
cd /nlp/scr/potsawee/workspace/soda-extension/marin
exec .venv/bin/python "experiments/audio/$SCRIPT" --run "$RUN"
