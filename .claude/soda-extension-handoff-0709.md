# SODA-Extension — Handoff (2026-07-09)

Fresh-session entry point. Supersedes `soda-extension-handoff-0708.md` (still
valid for environment facts, throughput findings, nlprun gotchas). The
experiment plan now lives IN THE REPO: read
`experiments/audio/EXPERIMENTS.md` (per-experiment goals/hypotheses/read-outs)
→ `experiments/audio/PROGRESS.local.md` (live status, untracked) →
`experiments/audio/DECISIONS.md` (design provenance).

## First 5 minutes

```bash
source /nlp/scr/potsawee/workspace/soda-extension/env.sh   # defines SODA_ROOT, MARIN_PREFIX (no $SODA!)
cd /nlp/scr/potsawee/workspace/soda-extension/marin        # branch soda-extension
squeue -u potsawee
cat experiments/audio/PROGRESS.local.md
```

## Status as of 2026-07-09 ~02:00

- **P1 headline pair DONE and EVALUATED** (both 3e18, d=768, dev-clean):
  | run | bits/audio-sec | bits/text-tok | sem NLL | acoustic NLL |
  |---|---|---|---|---|
  | p1-flat-uniform-b78282a0 | **562.58** | 1.933 | 2.696 | 3.58–4.41 |
  | p1-hier-moshi-765a7592 | 658.15 | **1.111** | **2.492** | 4.57–5.04 |
  Flat wins audio by 17%; hier wins text+semantic but loses every acoustic
  codebook — consistent with Moshi's alpha=100/100/1 starving acoustics, so
  **P1b (hier+uniform) is the decisive attribution run**. Eval JSONs
  committed: `experiments/audio/results/p1/`.
- **Campaign NOT yet launched** (user submits):
  `bash $SODA_ROOT/marin/experiments/audio/launchers/launch_campaign.sh`
  submits all 10 remaining runs in two tiers (P2+P1b → jagupard32–36 now;
  P4+P3+P1c → jagupard37–39 as Ada frees).
- Node policy (user, 2026-07-09): **jagupard37–39 preferred, 32–36 fallback**
  (supersedes 0708's "37–39 only"). Older jag nodes (19–31) are 24GB — never
  use; exclude lists must not name decommissioned nodes (sbatch errors).
- Eval-infra note: original eval job 16112798 was pinned to CPU-saturated
  jagupard39 → killed, fixed `$SODA`→`$SODA_ROOT` bug in `run_p1_eval.sh`,
  reran as 16112899 on jagupard33 (8 min on an Ampere A6000).
- Launchers now live **in-repo and canonical**: `experiments/audio/launchers/`
  (outside copies in `$SODA_ROOT/` deleted; `env.sh` deliberately stays
  outside and uncommitted — it references secrets).

## Unchanged hard constraints

No GCP / no `gs://`. Push code + upload artifacts after every unit of work.
Don't contact Marin maintainers. User submits long jobs; Claude may submit
short (<10 min) test jobs and held test jobs. Commits: user is author +
`Co-Authored-By: Claude <noreply@anthropic.com>`.

## Next after campaign launch

1. Babysit: first-hour loss-floor + MFU sanity per run (`data/runs/<run>.log`,
   W&B `soda-extension`). Resubmit any time-limit kill (resume is safe).
2. Eval every finished run (`launchers/run_p1_eval.sh` pattern → adapt
   checkpoint path/arm/d) → JSONs into `experiments/audio/results/`.
3. Analysis + writeup per `EXPERIMENTS.md` §Analysis; then package (Marin
   issue + PR from fork, HF upload of headline checkpoints + manifest).
