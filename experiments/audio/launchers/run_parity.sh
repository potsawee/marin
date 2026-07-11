#!/usr/bin/env bash
# HF-vs-JAX parity check for one exported run: run_parity.sh <run> [limit]
set -euo pipefail
RUN="$1"
LIMIT="${2:-16}"
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
cd /nlp/scr/potsawee/workspace/soda-extension/marin
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python -m experiments.audio.hf_export.verify_hf_parity --run "$RUN" --limit "$LIMIT"
