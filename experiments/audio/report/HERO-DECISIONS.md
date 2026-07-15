# Design decisions & rationale — SODA-Hier HERO run

The HERO analogue of `DECISIONS.md`: every non-obvious choice in the release
run, with justification and provenance, kept current as pilot/bench results
land. Tags follow `DECISIONS.md` ([PORTED] from the campaign recipe,
[NEW] reasoned choice without precedent, [DEVIATION] deliberate departure —
flagged for the writeup). Planning provenance: user decisions of 2026-07-12;
solver numbers from `isoflop_audio_target.solve_hier(budget, 1536,
depth_hidden=1024, depth_layers=4)`.

## The run in one line

One release-grade hierarchical model (**SODA-Hier**, ~1.09B params: ~990M
backbone + ~100M depth) trained on 6x RTX 6000 Ada for ~10–14 days on an
expanded ~350k-hour Yodas+Emilia corpus, ≤1 epoch, as the scaled extension of
the P1–P4 campaign and of the released flattened SODA line.

## Model

- **[NEW] Backbone d=1536 (L=15, 12 heads, head_dim 128, inter 6144, ~990M).**
  The solver's dims rules applied at the "~1B backbone" point — the CSM-tiny
  anchor (1B backbone + 100M decoder, sesame.com blog). Campaign support:
  width keeps paying on capabilities after NLL plateaus (P2; writeup §2.3).
- **[NEW] Depth dd=1152/L4 (~122M, 9 heads, head_dim 128).** The original
  CSM-match choice (dd1024, ~100M) is **incompatible with 6-GPU FSDP**:
  Levanter shards the depth transformer's embed axis and 1024 % 6 ≠ 0
  (MEASURED: bench 16138052 died with IndivisibleError on the
  (4,8,1,128,1024) depth attention param; backbone 1536 % 6 = 0, so 2/4-GPU
  runs passed). User chose dd1152 (2026-07-12) over dd960/~90M; shards on
  2/4/6/8. `depth_on_all_steps=True` charges the depth stack at every
  backbone step: **46.2% of per-step forward FLOPs**. Do not "optimize"
  this mid-run. P3 guardrail: under-sized depth costs TTS; past-knee depth
  is FLOPs-inefficient, not harmful. Total params **1111M**.
- **[PORTED] max_steps=1024** (≈82 s of audio per window; 1024 × 80 ms).
- **[PORTED] Vocab/tokenizer**: Mimi 12.5 Hz, 8 codebooks × 2048; unified
  head 130,308; full vocab 144,644; same `audio_vocab.py` id space.

## Loss weighting

- **[NEW, pilot-gated] Per-codebook geometric decay** `w_k = 100^(1−k/7)`:
  semantic (backbone) 100 → acoustic cb1..cb7 = (51.8, 26.8, 13.9, 7.2,
  3.7, 1.9, 1.0); text = 100. Motivation: the campaign 2×2 found Moshi
  100/100/1 wins likelihoods/ASR while uniform wins generation (p1b: TTS
  17.8 WER / SIM 0.330); decay is the principled midpoint (acoustic share of
  per-frame loss mass ≈ 51% vs uniform 87.5% / Moshi 6.5%). Precedent:
  VoiceCraft (arXiv:2403.16973) α=(5,1,0.5,0.1) over 4 EnCodec codebooks —
  "weighting the first residual codebooks more... leads to better
  performance"; their ablation mirrors our axis trade (intelligibility ↑,
  prosody ↓). Hierarchical precedents don't decay (Moshi flat 100:1,
  UniAudio uniform) — hence the pilot.
  - **Gate: pilot `p5-hier-decay100`** (3e18, d768, existing audio2 corpus,
    same eval battery) must land between p1-hier and p1b-hier on both axes.
    Fallback if it inherits the worst of both: uniform (p1b profile).
    **RESULT (2026-07-12): PASS — better than the gate required (a Pareto
    improvement over moshi), user chose decay.** Decay matches/beats moshi
    on EVERY semantic metric — ASR-0s 18.9 (campaign best, vs moshi 19.4 /
    uniform 22.5), tBLIMP 65.8 (best), tSC 63.1 (best), semantic NLL 2.485
    (best), sWUGGY 57.7 ≈ moshi 57.9 — while pulling generation off moshi
    toward uniform: TTS-WER 23.1 (moshi 29.6 → uniform 17.8), TTS-SIM 0.290
    (0.230 → 0.330), 100% termination, SALMon 67.9 (66.7 → 68.9). Not a
    mere midpoint: moshi-level semantics at zero cost + most of the
    moshi→uniform generation gap recovered. Full row in
    `results/campaign_results.csv` (p5-decay); parity OK.
- **[NEW] Implementation**: `AudioHierConfig.acoustic_weights:
  tuple[float,...] | None` (cb1..cb7; None → scalar `alpha_acoustic`;
  exact-equivalence and hand-computed-mean tests in `test_model_hier.py`).
  NOTE: adding the field changes `experiment_signature` hashes for ALL
  configs — old campaign run names would re-hash if ever relaunched (they
  are complete; their Levanter checkpoints were already deleted 2026-07-11).

## Compute & schedule

- **[PORTED] Budget convention** = 3× forward FLOPs (`audio_flops.py`),
  never 6ND.
- **[NEW] Budget set from measured step-time, not PLANNING_MFU.** The
  campaign's ~6–13% Ada MFU band is unmeasured at 1B; `PLANNING_MFU=0.30`
  is ~5× optimistic. Procedure: step-bench d1536/dd1024L4 at B=240 on 1 and
  6 GPUs → set `num_train_steps` to fill the wall-clock budget (target
  ~10–14 days ≈ 1.0–1.35e20; at 1.35e20: ~67k steps, 16.5B step-tokens).
  MEASURED: _pending_.
- **[DEVIATION] batch=240** (solver rule gives pow2 128/256). 240 divides
  4/6/8 GPUs → Levanter's batch-divisibility constraint is satisfied across
  every device count we might resume on. LR follows the ported rule at the
  actual batch: `0.33·√240/1536 = 0.00333`; `beta2 = 0.98^(240/128) =
  0.9629`.
- **[NEW] per_device_parallelism=10 (gradient accumulation ×4 at 6 GPUs)
  and a 2-GPU minimum.** MEASURED 2026-07-12 (benches 16137635/786/821):
  1-GPU OOMs regardless of microbatch (unsharded optimizer step ≈ 25G at
  1.09B — FSDP sharding is what makes the model fit, so ≥2 GPUs always);
  2-GPU at microbatch 20 still OOMs on a ~15G unified-head-CE temporary
  (the fused-CE autotune cache HIT reused block sizes tuned for the small
  campaign models — see open items). Microbatch 10 fits. pdp=10 divides
  B=240 at 4/6/8 GPUs (accum 6/4/3) and B=40 at the 2-GPU bench. Infra-only
  knob — excluded from the experiment hash.
- **[PORTED] Optimizer/schedule**: Cautious, wd 0.1, beta1 0.95, eps 1e-15,
  grad-clip 1, z-loss 1e-4, linear LR, warmup 0.1, decay 0.2,
  min_lr_ratio 0; precision p=f32/c=bf16; FSDP data-parallel default mesh.
  (Campaign lineage, NOT the released-4B lineage — wd 0.033/beta1 0.98 —
  so HERO stays comparable to P1–P4.)

## Data

- **[NEW] Corpus target ~400k hours = 150B flat tokens** (user-set
  2026-07-12, upsized from 131B after the measured 2-GPU bench showed the
  6-GPU run delivers ~1 epoch of 150B in ~11.7–13.4 days — 131B would have
  ended at ~10.2 days or repeated data). Mix [PORTED]: yodas .544 /
  emilia-yodas .303 / emilia .154, audio-only (transcripts supply text;
  Nemotron slice unavailable — known ~2pt tBLIMP cost, documented in
  EXPERIMENTS.md replication anchor).
- **[NEW] v3+v4 seeded picks** (seeds 45, 46; manifests in
  `soda-extension/data/`): yodas 25/38 EN dirs (86.5B avail); Emilia-YODAS
  935 files (47.9B); Emilia 787 files (24.3B) — each ≥ need at 150B with
  ~5% margin.
- **[NEW] Build to `$MARIN_PREFIX/audio3`** (audio2 kept until validated),
  arm_h only (no arm_f cache: −270G), 2‰ holdout unchanged,
  **chunk-parallel**: per-(source, chunk) TreeCache dirs as independent
  sub-sources whose mixture weights are set token-proportionally from chunk
  manifests; ~12 CPU-only Slurm jobs → ~2h wall-clock.
- Disk plan: peak ≈ 2.06T of 2.5T quota (82%); after audio2 deletion +
  purges ≈ 1.99T; +~91G checkpoints ≈ 83%.

## Ops

- **[NEW] Submission roles**: Claude submits/babysits everything except the
  HERO training job itself — that the user submits from a Claude-prepared,
  `--hold`-validated command. jag-standard's time cap is **14 days**
  (`-t 14-0`, confirmed 2026-07-12), so the ~11–19-day run needs **no job
  chain** — a single job covers most/all of it, and preemption/time-limit
  requeue resumes from checkpoint automatically (below). LAUNCHED
  2026-07-13: `nlprun -q jag -p standard -g 6 -c 16 -r 96G -n soda-hier-1b
  -t 14-0 -x <24G+down nodes>` on jagupard34 (6× Ampere, job 16143923;
  Ada was swarmed). Resume = rerun the same command (config-hash,
  node/GPU-count-safe).
- **[NEW] Checkpoints**: temporaries every 15 min + permanent every 10k
  steps (~13G each, ~130G total) + final. JAX checkpoints are KEPT (unlike
  the campaign) for resume/NLL-eval/export; prune only after release.
- **[NEW] Preemption is self-healing (verified 2026-07-13 on the live run,
  job 16143923)**: jag-standard `PreemptMode=REQUEUE` + the job's `Requeue=1`
  auto-requeue a preempted job (same id); it resumes from the latest
  checkpoint (temporaries every 15 min, permanents every 10k steps, stable
  config-hash path `audio2-runs/soda-hier-1b-08d907e0/checkpoints`) with
  <=15 min lost. A jag-urgent swarm (horatio) churns the cluster, so expect
  many preempt/resume cycles — no manual action needed per cycle. Manual
  relaunch (rerun the same nlprun) only on a full cancel or a deliberate
  migration to faster/idle hardware (resume is node/GPU-count-safe).
- **[PORTED] Monitoring**: W&B `soda-extension`; logs are ground truth
  (sacct COMPLETED unreliable; W&B resume high-water-mark quirk). Periodic
  holdout-NLL on checkpoints via 1-GPU `run_eval.sh` jobs.
- **[NEW] Slurm gotcha — MaxMemPerCPU=30000M on jag-standard**: a memory
  request above `cpus × 30000M` makes Slurm silently raise the allocated
  CPU count, and `srun` then dies at 0s with "cpus-per-task set by two
  different environment variables" (hit 5× on -c 8 -r 256G build retries,
  2026-07-12). Always request `-c ≥ ceil(mem/30000M)` — e.g. 256G → -c 9.
- **[NEW] Preprocessing task granularity = 1000-row slices, not whole
  files** (`_ROWS_PER_TASK`, 2026-07-12): at the HERO corpus's ~94%
  keep-rate one 11G single-row-group yodas parquet parses into ~30G of
  arrays per worker — whole-file tasks OOM-killed 128G jobs (16139472) and
  memory-thrashed co-resident chunks into their 6h time limit
  (16139423/25). Row-range tasks cap worker results at a few GB; verified
  byte-identical corpus output on the mini fixtures. Corrupt frame-layout
  docs (506 in the v4 shards) skip as DocParseError instead of killing the
  pool.

## Open items (updated as they resolve)

1. Pilot p5-hier-decay100 result + recipe decision — _pending_.
2. Measured step-time at 1B: 2-GPU Ada probe DONE 2026-07-12 (bench
   16137856, B=40, pdp=10, dd1024): **p50 MFU 10.63% vs 362 TF = 37.3
   eff-TFLOPS/GPU**, 4.5 s/step — ~1.7× the campaign band. 6-GPU B=240
   dd1152 bench DONE (16138080, 6× Ampere A6000 via sc-loprio preemption on
   jagupard35): **17.36 s/step p50 = 21.4 eff-TFLOPS/GPU → ~98% scaling
   efficiency** vs the Ada probe at the 1.7× device ratio (accum ×4
   amortizes all-reduce). **Projected 6× Ada: ~10.2 s/step → 96,517 steps
   ≈ 11.4 days; measured 6× Ampere fallback: 19.4 days.**
   **Provisional landing (dd1152): HERO_STEPS = corpus_steps/(240·1024) ≈
   96.5k = exactly 1 epoch of the 150B corpus; 2230 TF/opt-step → implied
   budget ≈ 2.15e20 ≈ 11.1 days at the measured 37.3 eff-TF/GPU (13.3 at a
   conservative 31)** — pin the exact step count from the audio3 aggregate
   manifest, and sanity-check with the 6-GPU bench before freeze.
3. Corpus build verification — DONE 2026-07-12: 12/12 chunks, aggregate
   mix-assertion passed (realized .546/.305/.149 vs target .544/.303/.154);
   **148.03B flat tokens = 23.356B backbone steps = 396k audio hours**.
   Build lessons folded into ops notes (row-group memory, MaxMemPerCPU,
   wipe-before-retry).
4. Config freeze: **soda-hier-1b-08d907e0** — HERO_STEPS=95,033 (exactly
   1 epoch), budget 2.12e20, ~11.2 days on 6× Ada / 19.1 on Ampere.
   Submit-command mechanics `--hold`-validated 2026-07-12 (job 16142486:
   6 GPU / 16 CPU / 96G / 7-day / Ada-only exclude list parsed correctly;
   cancelled) — the command is hash-independent, so this stands even if the
   pilot gate forces a weighting fallback (which would only re-hash the run
   name). Final freeze still awaits the pilot verdict.
5. Fused-CE block sizes for the d1536/130308 shape: the autotune cache HIT
   served block sizes tuned for the small campaign models and the batched_xla
   path allocated a ~15G temporary at microbatch 20. Worth clearing/re-running
   the autotune for the HERO shape (LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS only
   fires on a MISS) — potential memory and MFU win before launch; pdp=10 is
   the safe setting either way.
