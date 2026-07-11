#!/usr/bin/env bash
# Full preprocessing job for the SODA-extension caches. Submit via:
#   SLURM_CPU_BIND=none CONDA_PREFIX=unused nlprun -q jag -p standard -r 128G -c 8 -n soda-prep -t 0-8 \
#     'bash /nlp/scr/potsawee/workspace/soda-extension/marin/experiments/audio/launchers/run_preprocess.sh' \
#     -o /nlp/scr/potsawee/workspace/soda-extension/data/preprocess_full.log
set -euo pipefail
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
# caches are not appendable; always rebuild from scratch (fully regenerable)
rm -rf "$MARIN_PREFIX/audio2"
cd /nlp/scr/potsawee/workspace/soda-extension/marin
exec .venv/bin/python experiments/audio/preprocess_audio.py \
  --output "$MARIN_PREFIX/audio2" --token-budget 16e9 --workers 6
