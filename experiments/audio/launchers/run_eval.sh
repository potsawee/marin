#!/usr/bin/env bash
# Post-hoc NLL eval of a finished run: run_eval.sh <run-dir> <arm> <d> [step]
#   <run-dir> is the hashed run name, e.g. p4-flat-d768-1b5bb9e6.
#   [step] defaults to the latest step-N checkpoint under the run.
# Writes $SODA_ROOT/data/runs/<run-dir>.eval.json (bits/audio-second inside).
set -euo pipefail
RUN="$1"; ARM="$2"; D="$3"
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
CKROOT="$MARIN_PREFIX/audio2-runs/$RUN/checkpoints/$RUN"
STEP="${4:-$(ls "$CKROOT" | grep -E '^step-[0-9]+$' | sort -t- -k2 -n | tail -1)}"
cd /nlp/scr/potsawee/workspace/soda-extension/marin
.venv/bin/python experiments/audio/eval_audio_nll.py --arm "$ARM" --d "$D" \
  --checkpoint "$CKROOT/$STEP" \
  --output "$SODA_ROOT/data/runs/$RUN.eval.json"
echo "EVAL DONE: $RUN $STEP"
