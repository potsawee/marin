#!/usr/bin/env bash
# Export one run to HF format (CPU): run_export.sh <flat|hier> <run> [step]
#   run_export.sh flat p1-flat
#   run_export.sh hier p2-hier-d896 82998
set -euo pipefail
ARM="$1"
RUN="$2"
STEP="${3:-}"
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
cd /nlp/scr/potsawee/workspace/soda-extension/marin
case "$ARM" in
  flat) MODULE=experiments.audio.hf_export.export_flat_hf ;;
  hier) MODULE=experiments.audio.hf_export.convert_hier_to_hf ;;
  *) echo "arm must be flat or hier" >&2; exit 2 ;;
esac
exec env JAX_PLATFORMS=cpu .venv/bin/python -m "$MODULE" --run "$RUN" ${STEP:+--step "$STEP"}
