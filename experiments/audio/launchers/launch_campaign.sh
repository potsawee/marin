#!/usr/bin/env bash
# Launch the SODA-extension training campaign: 10 single-GPU runs, two tiers.
#
# Node policy (user, 2026-07-09): jagupard37-39 (RTX 6000 Ada) are the best
# GPUs and preferred; jagupard32-36 (RTX A6000 Ampere, also 48GB) are the
# fallback. At submission time 37-39 were saturated (0 free CPUs on 37/39;
# 38's free GPU fenced by a pending 4-GPU job), while 32-36 had ~5 free GPUs.
# So:
#   tier NOW  -> float across jagupard32-36: the runs we want moving
#                immediately (P2 frontier + P1b weighting decomposition).
#   tier ADA  -> float across jagupard37-39: queued, start as Ada GPUs free
#                (P4 anchors are short and backfill well; P3 + optional P1c).
# Within a tier, -x exclusion (not -m pinning) lets each job take the first
# GPU that frees on ANY tier node. Exclude lists may only name nodes that
# still exist; naming decommissioned ones fails sbatch ("Invalid node name").
#
# Wall-clock per 3e18 run: ~42h on Ada, up to ~2x on Ampere -> -t 4-0
# (P4 at 1e18: -t 2-0). A time-limit kill is safe: resubmit the same line and
# the run resumes from its last checkpoint (config-hash run names; validated).
# QOS jag-standard allows 16 GPUs / 16 running jobs per user.
set -euo pipefail

SODA=/nlp/scr/potsawee/workspace/soda-extension
LAUNCHERS=$SODA/marin/experiments/audio/launchers
OLD=jagupard19,jagupard20,jagupard26,jagupard27,jagupard28,jagupard29,jagupard30,jagupard31
AMPERE=jagupard32,jagupard33,jagupard34,jagupard35,jagupard36
ADA=jagupard37,jagupard38,jagupard39

submit() {
  local run="$1" script="$2" exclude="$3" tlimit="${4:-4-0}"
  SLURM_CPU_BIND=none CONDA_PREFIX=unused nlprun -q jag -p standard -g 1 -c 8 -r 40G \
    -n "$run" -t "$tlimit" -x "$exclude" \
    "bash $LAUNCHERS/run_train.sh $run $script" \
    -o "$SODA/data/runs/$run.log" 2>&1 | grep -E 'Submitted batch job|rror' || true
}

echo "== tier NOW: jagupard32-36 (free GPUs, start immediately) =="
submit p2-flat-d512 exp_isoflop_sweep.py    "$OLD,$ADA"
submit p2-hier-d512 exp_isoflop_sweep.py    "$OLD,$ADA"
submit p2-flat-d896 exp_isoflop_sweep.py    "$OLD,$ADA"
submit p2-hier-d896 exp_isoflop_sweep.py    "$OLD,$ADA"
submit p1b-hier     exp_isoflop_headline.py "$OLD,$ADA"

echo "== tier ADA: jagupard37-39 (queued; take Ada GPUs as they free) =="
submit p4-flat-d768 exp_isoflop_sweep.py    "$OLD,$AMPERE" 2-0
submit p4-hier-d768 exp_isoflop_sweep.py    "$OLD,$AMPERE" 2-0
submit p3-small     exp_depth_ablation.py   "$OLD,$AMPERE"
submit p3-large     exp_depth_ablation.py   "$OLD,$AMPERE"
submit p1c-flat     exp_isoflop_headline.py "$OLD,$AMPERE"

echo
squeue -u potsawee -o "%.10i %.14j %.2t %.11M %.11l %R"
