#!/usr/bin/env bash
# Chunk-parallel build of the HERO corpus (audio3, arm_h only, 131B tokens).
#
# Modes:
#   run_preprocess_hero.sh submit      submit all 12 chunk jobs via nlprun
#   run_preprocess_hero.sh chunk <source> <i/n>   (what each job executes)
#   run_preprocess_hero.sh aggregate   combine chunk manifests + assert mix
#
# Chunking: yodas 6 / emilia_yodas 4 / emilia 2. Each chunk is an independent
# sub-source cache (arm_h/{source}-c{i}); the keep threshold is computed over
# the FULL source, so chunking never changes the kept-document set.
set -euo pipefail
source /nlp/scr/potsawee/workspace/soda-extension/env.sh

MARIN=/nlp/scr/potsawee/workspace/soda-extension/marin
DATA=/nlp/scr/potsawee/workspace/soda-extension/data
OUT="$MARIN_PREFIX/audio3"
BUDGET=150e9
EXCLUDE=jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31

preprocess_args() {
  # 6 workers + 128G host RAM: the proven 16B-build ratio; 8 workers at 64G
  # OOM-looped on long v4 Emilia docs (see preprocess_audio.py batch_size note)
  echo "--output $OUT --token-budget $BUDGET --workers 6 --arms h \
    --yodas-pick $DATA/yodas-shard-pick-v4.json \
    --emilia-manifest $DATA/emilia-en-file-pick-v4.json"
}

case "${1:-}" in
submit)
  mkdir -p "$DATA/logs"
  for spec in yodas:6 emilia_yodas:4 emilia:2; do
    src="${spec%%:*}"; n="${spec##*:}"
    for ((i = 0; i < n; i++)); do
      SLURM_CPU_BIND=none CONDA_PREFIX=unused nlprun -q jag -p standard -g 0 -c 8 -r 128G \
        -n "prep-$src-$i" -t 0-6 -x "$EXCLUDE" \
        "bash $MARIN/experiments/audio/launchers/run_preprocess_hero.sh chunk $src $i/$n" \
        -o "$DATA/logs/prep-$src-c$i.log" | grep 'Submitted batch job'
    done
  done
  ;;
chunk)
  cd "$MARIN"
  # shellcheck disable=SC2046
  exec .venv/bin/python experiments/audio/preprocess_audio.py \
    $(preprocess_args) --source "$2" --chunk "$3"
  ;;
aggregate)
  cd "$MARIN"
  exec .venv/bin/python experiments/audio/preprocess_audio.py --output "$OUT" --aggregate
  ;;
*)
  echo "usage: $0 {submit|chunk <source> <i/n>|aggregate}" >&2
  exit 1
  ;;
esac
