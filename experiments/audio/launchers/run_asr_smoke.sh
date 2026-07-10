#!/usr/bin/env bash
# 10-item ASR 0-shot generation smoke through blueberry-eval (unmodified):
#   run_asr_smoke.sh <hf_checkpoint_dir> <output_dir>
# Uses the blueberry-soda-ext env; answers the trust_remote_code prompt via stdin.
set -euo pipefail
HF_DIR="$1"
OUT_DIR="$2"
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
cd /nlp/scr/potsawee/workspace/blueberry-eval/asr_librispeech
yes | /nlp/scr/potsawee/envs/blueberry-soda-ext/bin/python inference_0shot.py \
  --model_path "$HF_DIR" \
  --data_path /nlp/scr/potsawee/workspace/soda-extension/data/eval-smoke/dev-clean-10.json \
  --output_dir "$OUT_DIR" \
  --max_new_tokens 200 --temperature 0.0001 --top_p 0.8
echo "ASR SMOKE DONE: $HF_DIR"
