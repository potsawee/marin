# Experiment plan — flattened vs hierarchical audio LM campaign

The campaign registry: what each experiment (P1, P1b, P1c, P2, P3, P4) asks,
what is fixed, what varies, and how to read the result. Design rationale and
provenance for every shared choice lives in `DECISIONS.md`; file roles in
`README.md`; live run status in `PROGRESS.local.md` (untracked) and the
W&B project `soda-extension` (entity `potsawee`). **The findings narrative —
full result tables and interpretation across experiments — is
`RESEARCH-WRITEUP.md`; this registry and that write-up go hand in hand.**

## The research question

SODA flattens Mimi RVQ audio into one long token stream and trains a single
decoder on it. Reviewers asked: **is flattening the right factorization, or
would a CSM/Moshi-style hierarchical decomposition reach better audio NLL at
the same training compute?** The campaign answers this with matched-compute
(isoflop) pairs plus the ablations needed to attribute any gap.

Two positions the data could support:

- **Flattened (SODA)**: within-frame left-to-right attention over all 8
  codebooks is worth more than it costs; one head over a 144,644 vocab is a
  strong joint model.
- **Hierarchical (CSM/Moshi)**: spending sequence positions 8x more cheaply
  (one step per frame) frees backbone capacity for long-range structure; a
  small depth transformer suffices for within-frame acoustics, and inference
  is ~8x cheaper in KV length.

## Shared design (fixed across every run)

| dimension | choice |
|---|---|
| data | same 15.7B-token corpus, mix yodas .544 / emilia-yodas .303 / emilia .154, same 2‰ holdout (by base utterance id) |
| tokenization | Mimi 12.5 Hz, 8 codebooks x 2048 (1 semantic + 7 acoustic); Llama-3 BPE text; identical id space |
| architecture family | Qwen3 decoder blocks for everything (Arm F trunk, Arm H backbone + depth) |
| compute axis | budget = 3 x forward FLOPs (`audio_flops.py`), the SODA-paper convention — never naive 6ND |
| hyperparameters | batch, lr, beta2, steps all derived by `isoflop_audio_target.py`, regression-pinned to the old SODA sweep's rules (no per-run tuning) |
| optimizer | Cautious, z-loss 1e-4, bf16 — identical knobs both arms (see `DECISIONS.md`) |
| evaluation | teacher-forced NLL on LibriSpeech dev-clean (`eval_audio_nll.py`); headline = **bits per audio-second** (audio tokens only: semantic + acoustic); text reported separately as bits/text-token |
| seeds | one seed per config (single-seed campaign; noted as a limitation in the writeup) |

Both arms model the same joint distribution over the same documents
(`flat_tokens == steps + 7*frames`, asserted at preprocessing), so unweighted
held-out NLL is directly comparable. One run per (experiment, arm) cell; a run
is one `--run` name in an `exp_*.py` script, launched via
`launchers/run_train.sh <run> <script>`.

## Campaign at a glance

| exp | runs | budget | varies | everything else |
|---|---|---|---|---|
| P1 | p1-flat, p1-hier | 3e18 | architecture (at each arm's home loss recipe) | d=768 |
| P1b | p1b-hier | 3e18 | Arm H loss weighting → uniform | d=768 |
| P1c | p1c-flat (optional) | 3e18 | Arm F loss weighting → Moshi | d=768 |
| P2 | p2-{flat,hier}-d{512,896} | 3e18 | width d ∈ {512, 896} | home recipes |
| P3 | p3-small, p3-large | 3e18 | Arm H depth-transformer allocation | backbone d=768 |
| P4 | p4-{flat,hier}-d768 | 1e18 | compute budget | d=768, home recipes |

Exact per-run configs (params, batch, steps, lr): run
`uv run python experiments/audio/isoflop_audio_target.py`.

---

## P1 — the headline pair (`exp_isoflop_headline.py`)

- **Question.** At 3e18 FLOPs and d=768, which architecture reaches better
  held-out audio NLL, each trained with its published loss recipe?
- **Fixed.** Budget, width, data, everything in the shared table.
- **Varies.** Architecture: Arm F (uniform CE, the published SODA recipe) vs
  Arm H (Moshi-weighted CE, alpha 100 text/semantic, 1 acoustic).
- **Hypothesis.** SODA's position: flat wins bits/audio-second at matched
  compute. Moshi's recipe should nonetheless give Arm H much better text and
  semantic NLL (alpha=100 concentrates capacity there).
- **Caveat by construction.** P1 compares *architecture+recipe bundles*, not
  architectures alone — that attribution is exactly what P1b/P1c add.
- **Result (2026-07-09).** Flat **562.6** vs hier **658.1** bits/audio-second
  (flat wins audio by 17%); hier wins text (1.111 vs 1.933 bits/text-token)
  and semantic NLL (2.492 vs 2.696 nats) but loses every acoustic codebook
  (4.57–5.04 vs 3.58–4.41 nats) — the deficit sits exactly where Moshi's
  weight is 1/100th, so P1b is the decisive next read.
  JSONs: `results/p1/`.

## P1b — hier, uniform loss (`exp_isoflop_headline.py`)

- **Question.** How much of hier's P1 audio deficit is the loss *weighting*
  rather than the *architecture*?
- **Fixed.** Everything in p1-hier (same 308M model, batch, lr, steps).
- **Varies.** Loss weighting only: Moshi (100/100/1) → uniform.
- **Read-out.** Completes the architecture arm of the 2x2:
  - If p1b-hier ≈ p1-flat on bits/audio-second → the P1 gap was weighting;
    "flattening wins" would be overclaiming — the honest claim becomes
    "uniform CE wins, factorization is secondary".
  - If p1b-hier stays well above p1-flat → the gap is architectural; the
    flattened factorization itself is compute-efficient.
  - Expect p1b to trade away the text/semantic advantage; report both axes.
- **Result (2026-07-10).** p1b-hier **631.8** bits/audio-second vs p1-hier
  658.2 and p1-flat 562.6: uniform weighting recovers only **28%** of the P1
  gap — **~72% is architectural**. Costs: text 1.111 → 1.341 bits/text-token
  (still well ahead of flat's 1.933), semantic 2.492 → 2.547. Per-codebook
  acoustics improve across the board (ac1 4.57 → 4.40, ac7 4.94 → 4.71) but
  stay far above flat (3.58 / 4.41). JSON: `results/p1b/`.

## P1c — flat, Moshi weighting (optional; `exp_isoflop_headline.py`)

- **Question.** Symmetric cell completing the 2x2 (architecture x weighting):
  does Moshi weighting damage flat's audio NLL the way it appears to damage
  hier's, and does it buy flat the same text NLL win?
- **Fixed.** Everything in p1-flat.
- **Varies.** Loss weighting only: uniform → Moshi (100/100/1).
- **Read-out.** With P1/P1b it gives main effects and the interaction term of
  {arch} x {weighting} on both metrics. Optional: run last; skip only if GPUs
  are scarce.
- **Result (2026-07-10).** p1c-flat **653.4** bits/audio-second — Moshi
  weighting costs *flat* 91 of the 96-bit P1 gap, while it costs *hier* only
  26 (658.2 vs p1b's 631.8). The 2x2 is strongly non-additive (interaction
  ≈ −65 bits/audio-second): **the depth factorization and the loss weighting
  are partially redundant ways of de-prioritizing acoustics.** On
  capabilities the same diagonal is monotone in the other direction — ASR-0s
  37.1 (flat-unif) → 28.5 (flat-moshi) → 22.5 (hier-unif) → 19.4 (hier-moshi)
  — and tBLIMP shows a floor: all cells except flat-uniform tie at ~64.5%.
  JSON: `results/p1c/`; interpretation: RESEARCH-WRITEUP.md §2.2.

## P2 — width sweep at 3e18 (`exp_isoflop_sweep.py`)

- **Question.** Is the P1 conclusion an artifact of d=768, and where is each
  arm's compute-optimal width at 3e18?
- **Fixed.** Budget 3e18, home loss recipes (flat uniform / hier Moshi), the
  solver's batch/lr/steps rules.
- **Varies.** Width d ∈ {512, 896} per arm (d=768 comes from P1), i.e. model
  size vs tokens along the isoflop contour: flat 173M/298M/375M,
  hier 178M/308M/389M.
- **Hypothesis.** Each arm has an interior optimum; the flat-vs-hier ordering
  is stable across widths (if it flips with width, the headline claim must be
  stated per-width — that itself is a finding).
- **Read-out.** Per-arm isoflop curve (bits/audio-second vs d) at 3e18; the
  minimum per arm is that arm's frontier point for the headline figure.
- **Result (2026-07-10, complete).** Both isoflop curves are **plateaus, not
  peaks** — flat: 567.7 / 562.6 / 562.2 and hier: 668.1 / 658.2 / 657.2
  bits/audio-second at d512/768/896. The flat-vs-hier NLL ordering is stable
  at every width (~17%). Capabilities are NOT width-flat, though:
  p2-hier-d896 is the campaign's best model (ASR 16.1/15.1 WER), while
  p2-flat-d512's ASR collapses (46.9 0-shot, 49.6 2-shot — the only 2-shot
  inversion in the campaign; plausibly its 4096-token window straining under
  two-shot audio prompts, which hier's 1024-step window never feels).
  JSONs: `results/p2/`; interpretation: RESEARCH-WRITEUP.md §2.3.
- **Epoch caveat (writeup must flag).** p2-hier-d512 is the campaign's only
  multi-epoch run: 3.68B backbone steps needed vs 2.48B in the corpus =
  **1.48 epochs** (the hier stream is ~6.3x shorter than the flat one for the
  same documents, and the solver gives small models more data). Every other
  run is sub-epoch: flat ≤0.26 everywhere; hier d768 0.75, d896 0.56,
  p3-small 0.96, p3-large 0.51, p4 0.25. The loader wraps cleanly (levanter
  MixtureDataset restart strategy, modulo indexing). Bias direction is
  conservative: repeated data is slightly worse than fresh at ~1.5 epochs, so
  it can only understate hier at d512, not manufacture a hier win. Holdout is
  untouched by repetition (split by base-utterance-id at preprocessing).
  *Empirical closure:* p2-hier-d512 sits exactly on the hier width trend on
  every metric — the 1.48 epochs visibly cost nothing.

## P3 — depth-transformer allocation (Arm H only; `exp_depth_ablation.py`)

- **Question.** At fixed total budget, does it matter how much of Arm H's
  compute lives in the depth transformer (the within-frame acoustic model)?
  Addresses "you chose the hierarchical baseline's split badly" reviews.
- **Fixed.** Budget 3e18, backbone d=768, Moshi weighting, solver rules
  (steps rebalance automatically as depth size changes).
- **Varies.** Depth transformer: dd256L2 (~3% of params, p3-small) vs
  dd384L4 (P1 default, ~30% of per-step forward FLOPs) vs dd512L6 (~13% of
  params, p3-large).
- **Hypothesis.** CSM/Moshi lore: small depth suffices — acoustic NLL is
  insensitive above a modest size, so p3-small ≈ p1-hier, and enlarging depth
  (p3-large) wastes budget the backbone could use.
- **Read-out.** bits/audio-second (and its per-codebook split) vs depth
  allocation; strengthens or bounds the Arm-H side of the headline claim.
- **Result (2026-07-10).** The default allocation sits at the knee:
  p3-small (dd256L2) **673.1** vs p1-hier (dd384L4) 658.2 vs p3-large
  (dd512L6) **656.8** bits/audio-second — shrinking depth costs 15,
  growing it buys 1.4. Capabilities are nearly insensitive to depth size
  (p3-small even posts the best hier sWUGGY-sem 64.7%, sSC 53.0%, tBLIMP
  65.6%) — everything the benchmarks measure lives in the backbone/semantic
  stream. The "you sized the hier baseline's depth badly" review is answered.
  JSONs: `results/p3/`; interpretation: RESEARCH-WRITEUP.md §2.4.

## P4 — 1e18 anchors (`exp_isoflop_sweep.py`)

- **Question.** How does the flat-vs-hier gap move with compute? Two budgets
  per arm give the first frontier slope — "which factorization wins" could be
  budget-dependent.
- **Fixed.** d=768, home recipes, solver rules.
- **Varies.** Budget: 1e18 (vs P1's 3e18). Flat 298M/21442 steps;
  hier 308M/75429 steps.
- **Hypothesis.** Gap direction is stable in budget; if it narrows with
  compute, extrapolation beyond 3e18 must be hedged in the writeup.
- **Read-out.** Per-arm two-point frontier (bits/audio-second vs budget) at
  fixed width; combined with P2 this is the isoflop-frontier figure.
- **Result (2026-07-10).** 1e18 anchors: flat **586.5**, hier **674.5**
  bits/audio-second — flat ahead by 15.0% at 1e18 vs 17.0% at 3e18, and flat
  gains more per budget tripling (−23.9 vs −16.4 bits/audio-second). The gap
  is stable-to-widening with compute; no sign of a crossover. Text improves
  steeply with budget for flat (2.642 → 1.933 bits/text-token).
  JSONs: `results/p4/`.

---

## Analysis phase (no training)

1. **Isoflop frontier figure.** bits/audio-second vs compute per arm
   (P1 + P2 + P4), minima marked.
2. **Weighting 2x2.** P1 x P1b (x P1c) decomposition of architecture vs
   weighting on both audio and text axes.
3. **Per-codebook NLL.** Semantic vs acoustic split per arm (already emitted
   by `eval_audio_nll.py`).
4. **Depth allocation.** P3 curve.
5. **Inference economics (analytic).** Flat: 8 x fwd(N_F) per audio-second and
   KV length 8T; hier: fwd(N_backbone) + 7 x fwd(N_depth) and KV length T.
   Uses `audio_flops.py` accountants; no new runs.

## Capability results (blueberry-eval suite, 2026-07-10)

All 12 runs evaluated end-to-end through blueberry-eval (branch
`soda-extension`, env `blueberry-soda-ext`): ASR 0/2-shot WER (LibriSpeech
test-clean), paired-likelihood tasks in TWO scoring variants (all-tokens
uniform and semantic-only), text tasks, and the blueberry NLL battery.
**Canonical numbers: `results/campaign_results.csv`; the full table and all
interpretation live in `RESEARCH-WRITEUP.md` Part 2** (this registry does
not duplicate them). Registry-level capsule:

- Hierarchical+Moshi wins ASR (best: p2-hier-d896, 16.1/15.1 WER) and every
  above-chance semantic/lexical/text task; flattened wins the acoustic axis
  (uniform-scored SALMon + all acoustic-NLL components). See
  RESEARCH-WRITEUP.md §2.1 and Part 1 for why a single scalar NLL cannot
  arbitrate between the arms (measure-dependence).
- The semantic-only scoring variant matters: sWUGGY +6.4 points mean,
  SALMon −3.2 (and SALMon's arm ranking flips under semantic scoring);
  sBLIMP/sSC are at chance at these budgets under both variants.
- Replication anchor vs the old-SODA twin of p1-flat
  (Nemotron-inclusive mix): likelihood tasks reproduce within ~1 point;
  tBLIMP dips (62.0 vs 64.2, less text data) and ASR improves sharply
  (37.1 vs 58.1, more paired audio per token) — a data-mix effect.

## HF export + capability-eval bridge

Any run (and any future HERO run registered in an exp script's `RUNS`) can be
exported to a HuggingFace checkpoint under
`$MARIN_PREFIX/audio2-runs/<run_id>/hf/step-<N>/` — the layout blueberry-eval
consumes. Exports are user-run, on CPU:

```bash
bash experiments/audio/launchers/run_export.sh flat p1-flat        # Qwen3ForCausalLM
bash experiments/audio/launchers/run_export.sh hier p1-hier       # SodaHierForCausalLM (trust_remote_code)
bash experiments/audio/launchers/run_parity.sh p1-flat 16         # HF-vs-JAX NLL parity (1 GPU, ~5 min)
```

- Arm F exports as a stock `Qwen3ForCausalLM` (QK-norm preserved; config
  down-converted to legacy `rope_theta`/`rope_scaling` keys so transformers
  4.x loads it correctly). The exported tokenizer auto-prepends BOS.
- Arm H exports as `SodaHierForCausalLM` — a trust_remote_code torch port
  (`hf_export/modeling_soda_hier.py`, copied into each export) that exposes
  the exact factorized conditionals over the flat interleaved stream via
  `forward` and two-stage sampling via `generate`, so blueberry-eval treats
  both arms identically. Bit-parity vs the JAX model is enforced by
  `test_hf_hier.py` (unit, both head-dim geometries) and
  `hf_export/verify_hf_parity.py` (per-checkpoint).
- Capability evals run in the `blueberry-soda-ext` conda env (definition:
  `requirements-blueberry-soda-ext.txt` on blueberry-eval's `soda-extension`
  branch). All blueberry loaders (model + tokenizer + lm-eval `--model_args`)
  pass `trust_remote_code=True`, so hier checkpoints run with no prompt; see
  blueberry's `README-steps-soda-extension.md` for the full workflow.

## Status and artifacts

**Campaign complete (2026-07-10): all 12 runs trained to their final step,
NLL-evaluated, HF-exported, and capability-evaluated.** Remaining phases:
analysis figures, inference-economics, writeup, HF release (user-curated).

- Every run logs to W&B project `soda-extension`; run names carry the config
  hash (resume-safe). Checkpoints:
  `$MARIN_PREFIX/audio2-runs/<run>/checkpoints/<run>/step-N` + verified
  `hf/step-N` export per run.
- NLL eval JSONs: `experiments/audio/results/p{1,1b,1c,2,3,4}/`; combined
  NLL+capability matrix: `experiments/audio/results/campaign_results.csv`.
- Raw capability outputs (per-sample logs, transcripts, stats):
  `blueberry-eval/auto_evals/soda-extension/<run_id>-step<N>/`.
- Known holes, accepted: TTS WER/SIM not run (tts1/tts2 excluded from `all`);
  HellaSwag/MMLU absent for hier checkpoints (old-transformers lm-eval env
  can't read the 5.x-saved tokenizer_class) and expected ≈chance at these
  budgets anyway.
