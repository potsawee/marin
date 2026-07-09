#!/usr/bin/env bash
set -euo pipefail
source /nlp/scr/potsawee/workspace/soda-extension/env.sh
cd /nlp/scr/potsawee/workspace/soda-extension/marin
CK=$MARIN_PREFIX/audio2-runs
echo "########## P1-FLAT eval ##########"
.venv/bin/python experiments/audio/eval_audio_nll.py --arm flat --d 768 \
  --checkpoint $CK/p1-flat-uniform-b78282a0/checkpoints/p1-flat-uniform-b78282a0/step-64325 \
  --output $SODA_ROOT/data/runs/p1-flat.eval.json
echo "########## P1-HIER eval ##########"
.venv/bin/python experiments/audio/eval_audio_nll.py --arm hier --d 768 \
  --checkpoint $CK/p1-hier-moshi-765a7592/checkpoints/p1-hier-moshi-765a7592/step-56570 \
  --output $SODA_ROOT/data/runs/p1-hier.eval.json
echo "########## HEADLINE ##########"
.venv/bin/python - <<'PY'
import json
for a in ("flat","hier"):
    d=json.load(open(f"/nlp/scr/potsawee/workspace/soda-extension/data/runs/p1-{a}.eval.json"))
    print(f"{a}: bits/audio-sec={d['bits_per_audio_second']:.3f}  bits/text-tok={d.get('bits_per_text_token')}  nll/total={d['nll/total']:.0f}  docs={d['docs']}")
PY
echo "P1 EVAL DONE"
