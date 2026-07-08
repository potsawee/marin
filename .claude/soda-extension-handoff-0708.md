# SODA-Extension — Handoff (2026-07-08)

Fresh-session entry point. Supersedes `soda-extension-handoff-0707.md` (that one
covers Phase-1 cluster bring-up, still valid for environment facts). Read this
first, then the in-repo docs it points to.

## 0. First 5 minutes of a new session

```bash
source /nlp/scr/potsawee/workspace/soda-extension/env.sh          # MUST run first (MARIN_PREFIX, caches, secrets)
cd /nlp/scr/potsawee/workspace/soda-extension/marin               # branch: soda-extension (fork potsawee/marin, SSH remote)
squeue -u potsawee                                                # are P1 (or later) jobs still running?
cat experiments/audio/PROGRESS.local.md                           # live status tracker (untracked)
```
Then read, in the repo: `experiments/audio/README.md` (what each file is),
`experiments/audio/DECISIONS.md` (every design decision + provenance, for the
writeup), `experiments/audio/findings/` (investigations: MFU/throughput, the
0.0 bug). All committed and pushed as of this handoff (HEAD `e238a1244`).

## 1. The project (one paragraph)

Compare two audio-LM architectures at **matched training compute** (isoflop),
same data, on Stanford NLP RTX 6000 Ada GPUs, to answer reviewers' "is flattened
the right factorization?":
- **Arm F (flattened)** = SODA: one Qwen3 decoder over the flat 144,644-vocab
  Mimi token stream (1 semantic + 7 acoustic codebooks/frame, 8 tokens/frame),
  seq 4096. Uniform CE (published SODA recipe).
- **Arm H (hierarchical, CSM/Moshi-style)** = Qwen3 **backbone** over "steps"
  (one text/special token, or one audio frame = sum of its 8 codebook embeddings;
  unified 130,308-way head predicts the next step's text/special/semantic token)
  + a small Qwen3 **depth** transformer over the 8 codebook slots predicting the
  7 acoustics. seq 1024 steps. Moshi-weighted CE (alpha=100 text+semantic, 1 acoustic).

Both model the same joint (`flat_tokens == steps + 7*frames` per doc, asserted),
so held-out NLL is comparable. **Headline metric = bits per audio-second** on
LibriSpeech dev-clean. Deliverable: isoflop frontier per arm + depth-allocation
(P3) and loss-weighting (P1b/P1c) ablations + inference-economics, packaged as a
Marin-house issue/PR from the fork.

## 2. Hard constraints (unchanged, do not violate)

- **No GCP / no `gs://` ever** (Stanford identity deprovisioned).
- **Push code to GitHub + upload artifacts to HF after every unit of work** —
  nothing of value lives only on this machine (borrowed-time access).
- **Do NOT contact Marin maintainers** or request cluster access.
- **User submits all long-running jobs**; Claude may submit short (<10 min) test
  jobs. **GPU jobs on jagupard37–39 only.** Old `/nlp/scr/potsawee/workspace/marin-audio`
  is read-only reference.
- **Commits:** user is author; add `Co-Authored-By: Claude <noreply@anthropic.com>`.

## 3. Current status (as of 2026-07-08 ~07:20)

**P1 headline pair RUNNING on 4 GPUs each, healthy, descending:**
| job | run (wandb id) | node | at handoff | ETA |
|---|---|---|---|---|
| 16104481 | `p1-flat-uniform-b78282a0` (flat, B=8, 64326 steps, lr 0.00122) | jagupard37 | step ~1950, loss 5.3 | ~16h |
| 16104482 | `p1-hier-moshi-765a7592` (hier, B=32, 56571 steps, lr 0.00243) | jagupard38 | step ~1880, loss 3.1 | ~15h |

- W&B: project **`soda-extension`**, entity `potsawee`. Loss floors were ~11.9
  (flat) / ~11.8 (hier); both descending correctly.
- Logs: `soda-extension/data/runs/p1-{flat,hier}.log`.
- Checkpoints: `$MARIN_PREFIX/audio2-runs/<run-name>/checkpoints/` (+ hf/ export).
- **Check on resume:** `squeue -u potsawee`; if gone, `sacct -j <id>` + tail the
  log; final checkpoint under the run dir means it finished.

**Data (done, validated):** `$MARIN_PREFIX/audio2/` = 15.7B train tokens across
arm_{f,h}/{yodas,emilia_yodas,emilia}/{train,holdout}, mix .544/.303/.154, 2‰
holdout by base-utterance-id hash. All arm_f splits token-exact vs manifest.
Corpus def (10 seeded YODAS-en shards + Emilia EN v2 manifest) in
`preprocess_audio.py` + `data/emilia-en-file-pick-v2.json`.

## 4. Key facts & gotchas (carry these forward)

- **Mimi frame = 8 tokens** (1 sem + 7 acoustic), codebook 2048, 12.5 Hz.
  Vocab 144,644 = Llama3 BPE 128256 + 4 specials (128256 text_start … 128259
  audio_end) + audio ids `128260 + cb*2048 + idx`. Unified backbone head = 130,308.
- **Compute axis = 3 x forward FLOPs** (SODA-paper convention; the old sweep's
  internal budgets were forward-only, relabelled x3). Never naive-6ND.
- **THROUGHPUT (findings/mfu-throughput.md):** these small models run at **~6% MFU**
  (compute-bound: d=768 + 144k-vocab head on Ada GDDR6). NOT fixable by data
  loader, attention (flash saves only ~8%), or batch size. **Consequence: run
  campaign configs at 1 GPU each, parallelized** — ~42h/run but ~1.6x more
  GPU-hour-efficient and better campaign throughput than 4-GPU/run (~17h but 68
  GPU-h). P1 is on 4 GPUs only because it launched before this was known; let it finish.
- **Run names carry a config hash** (`experiment_signature`, Marin-canonical
  `fingerprint_hash(canonical_json)`, 8 hex). Infra-independent (device count/
  paths excluded) → same config resumes across GPU counts; different config →
  different dir, no stale-resume. This fixed a real bug (see findings/).
- **nlprun gotchas:** needs `CONDA_PREFIX=unused` (reads the var, unused by us)
  and a **bash wrapper script** (its launcher is /bin/sh; `source` fails). Use
  `SLURM_CPU_BIND=none` for multi-worker jobs. Env.sh is POSIX-safe.
- **z-loss = 1e-4** on both arms (a deviation from the sweep's none; applied to
  both so no confound — see DECISIONS.md). Real runs keep fused-CE autotune ON
  (~7 min one-time compile); dev iteration can set
  `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` + `XLA_FLAGS=--xla_gpu_autotune_level=0`.

## 5. What's next (in order)

1. **When P1 finishes: eval both checkpoints** and get the first comparison:
   ```bash
   uv run python experiments/audio/eval_audio_nll.py --arm flat --d 768 \
     --checkpoint $MARIN_PREFIX/audio2-runs/p1-flat-uniform-b78282a0/checkpoints/step-64325 \
     --output data/runs/p1-flat.eval.json
   uv run python experiments/audio/eval_audio_nll.py --arm hier --d 768 \
     --checkpoint $MARIN_PREFIX/audio2-runs/p1-hier-moshi-765a7592/checkpoints/step-56570 \
     --output data/runs/p1-hier.eval.json
   ```
   Compare `bits_per_audio_second` (headline) + per-codebook splits. (Confirm the
   exact final step-dir name by listing the checkpoints dir.)
2. **Run the rest of the campaign — 1 GPU per run, parallelized** across
   jagupard37–39. Launch pattern (user submits):
   ```bash
   SLURM_CPU_BIND=none CONDA_PREFIX=unused nlprun -q jag -p standard -g 1 -r 40G -c 8 \
     -n <RUN> -t 1-0 -m jagupard39 \
     'bash /nlp/scr/potsawee/workspace/soda-extension/run_train.sh <RUN> <EXP_SCRIPT>' \
     -o /nlp/scr/potsawee/workspace/soda-extension/data/runs/<RUN>.log
   ```
   Runs (RUN, EXP_SCRIPT):
   - P2 sweep + P4 anchors → `exp_isoflop_sweep.py`: `p2-flat-d512`, `p2-hier-d512`,
     `p2-flat-d896`, `p2-hier-d896`, `p4-flat-d768`, `p4-hier-d768`
   - P3 depth ablation → `exp_depth_ablation.py`: `p3-small`, `p3-large`
   - P1b/P1c weighting → `exp_isoflop_headline.py` (default, omit script arg):
     `p1b-hier`, `p1c-flat`
   Inspect the full matrix + sizes: `uv run python experiments/audio/isoflop_audio_target.py`.
3. **Analysis:** isoflop frontier (bits/audio-second vs compute per arm); per-codebook
   NLL; weighting decomposition (P1 vs P1b, +P1c for the 2x2); depth-allocation (P3).
4. **Inference-economics:** flattened 8*fwd(N_F)/audio-sec vs hier backbone+7*depth;
   KV length T vs 8T.
5. **Package:** Marin-house experiment issue + PR from the fork; upload headline
   checkpoints (hf/ export) + holdout manifest to a private HF repo.

## 6. Key paths

- Repo/branch: `/nlp/scr/potsawee/workspace/soda-extension/marin`, `soda-extension`.
- Code: `experiments/audio/` (see README.md). Launchers:
  `soda-extension/{run_train.sh, run_preprocess.sh, run_smoke_gpu.sh}`.
- Data caches: `$MARIN_PREFIX/audio2/…`; run outputs: `$MARIN_PREFIX/audio2-runs/<run>/`.
  (`MARIN_PREFIX=/nlp/scr/potsawee/workspace/soda-extension/marin-store`.)
- Eval set: `soda-extension/data/librispeech-mm-eval/data/dev_clean_asr-*.parquet`.
- Trackers (untracked, local): `experiments/audio/{PROGRESS,SMOKE_LADDER}.local.md`.
- Everything smoke-validated (rungs 0–6 PASS); multi-GPU data-parallel verified correct.
