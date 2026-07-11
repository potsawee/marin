#!/usr/bin/env bash
# Post-hoc NLL eval of a finished run: run_eval.sh <run-dir> <arm> <d> [step] [extra args...]
#   <run-dir> is the hashed run name, e.g. p4-flat-d768-1b5bb9e6.
#   [step] defaults to the latest step-N checkpoint (pass "latest" to skip it
#   when extra args follow). Extra args go to eval_audio_nll.py verbatim —
#   needed for non-default hier depth geometry, e.g. the P3 ablations:
#     run_eval.sh p3-hier-dd256L2-<hash> hier 768 latest --depth-hidden 256 --depth-layers 2
# Writes $SODA_ROOT/data/runs/<run-dir>.eval.json (bits/audio-second inside).
set -euo pipefail
RUN="$1"; ARM="$2"; D="$3"
shift 3
STEP="latest"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  STEP="$1"
  shift
fi
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
CKROOT="$MARIN_PREFIX/audio2-runs/$RUN/checkpoints/$RUN"
if [ "$STEP" = "latest" ]; then
  STEP=$(ls "$CKROOT" | grep -E '^step-[0-9]+$' | sort -t- -k2 -n | tail -1)
else
  STEP="step-$STEP"
fi
cd /nlp/scr/potsawee/workspace/soda-extension/marin
.venv/bin/python experiments/audio/eval_audio_nll.py --arm "$ARM" --d "$D" \
  --checkpoint "$CKROOT/$STEP" \
  --output "$SODA_ROOT/data/runs/$RUN.eval.json" "$@"
echo "EVAL DONE: $RUN $STEP"
