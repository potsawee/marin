# Design decisions & rationale — flattened vs hierarchical audio LM

A record of every non-obvious decision in this comparison, with justification
and provenance, so the eventual writeup can state exactly what was done and why.
Each item is tagged:

- **[PORTED]** — taken verbatim from the released SODA isoflop sweep
  (`marin-audio/experiments/audio/isoflop_audio_sweep.py`), regression-pinned in
  `test_audio_flops.py::test_solver_reproduces_old_sweep_grid_point`.
- **[NEW]** — a reasoned choice with no old-sweep precedent (the two-transformer
  arm is novel); user-approved.
- **[DEVIATION]** — differs from the old sweep on purpose; flagged for the writeup.

## The comparison in one line

Two architectures at matched training compute over identical Mimi-RVQ audio+text
data: **Arm F** (SODA flattened decoder-only) vs **Arm H** (CSM/Moshi-style
backbone + per-frame depth transformer). Headline metric: held-out **bits per
audio-second** (factorization-agnostic). Deliverable: isoflop frontier per arm +
depth-allocation and loss-weighting ablations + inference-economics.

## Data

- **[NEW] S+A+T scope.** Train on the utterance-level interleaved text+audio
  documents as shipped in `soda-research/{yodas2,emilia}-mm-pretrain`. Rationale:
  the data already carries transcripts; the CSM backbone models the combined
  {S+A} frame stream interleaved with text tokens. No Nemotron text-only data is
  available anymore, so there is no pure-text component.
- **[NEW] Corpus = 10 seeded YODAS-en shard dirs + seeded Emilia-EN slice**
  (manifests in `soda-extension/data/`, shard list in `preprocess_audio.py`).
  Seeded random selection (not en000..en009) to avoid channel/ordering bias from
  a contiguous block.
- **[PORTED] Mixture ratios** yodas : emilia-yodas : emilia = 0.544 : 0.303 :
  0.154 — the old SODA English speech mix (131:73:37B tokens) renormalized after
  dropping the unavailable 5% Nemotron text.
- **[NEW] Corpus size = 16B tokens** (mix-preserving maximum of on-disk data;
  Emilia is the binding source). Rationale: hold Arm H's data repetition low —
  0.75 epochs at the d=768 headline, 1.49 at d=512 (Arm F ≤0.26 everywhere).
  Expanded from an initial 5B build once the epoch asymmetry (below) was measured.
- **[NEW] Holdout = 2 per-mille by hash of the *base* utterance id.** The
  `_type1`/`_type2` interleave-order twins share a base id, so they always land
  on the same split side — no train/test leakage across interleavings. Identical
  doc set for both arms by construction.
- **[NEW] `emilia-mm-pretrain-fix`** (whitespace-corrected re-release) over the
  original; both arms see the same corrected text.

## Compute accounting

- **[PORTED+RECONCILED] Budget = 3 x detailed forward FLOPs.** The old sweep's
  internal budgets counted forward FLOPs only (a known bug: backprop unaccounted);
  the SODA paper relabelled all numbers x3. Our `TRAIN_MULT=3` therefore plots
  directly onto the paper's compute axis — paper-3e18 reproduces the old grid's
  d=768/298M point (2.11B tokens). **Never use naive 6ND** (it overestimates ~25%
  for these embedding-heavy small models); anchor on fitted N*, let the accountant
  set D. Arm H charges the depth transformer at *every* backbone step, matching
  the `depth_on_all_steps=True` implementation exactly.

## Architecture — Arm H (the novel arm)

- **[NEW] Backbone = the same Qwen3 stack as Arm F**, over "steps" (one text/
  special token, or one audio frame = sum of its 8 codebook embeddings). A unified
  head over text + specials + semantic codebook (130,308) predicts the next step's
  primary token. Reusing the Qwen3/Llama blocks keeps numerics aligned with Arm F
  and gives RoPE/RMSNorm/GQA/checkpointing for free.
- **[NEW] Depth transformer** = a small Qwen3 stack whose sequence axis is the
  8-codebook slot axis, batched over steps; per-codebook 2048-way heads predict
  acoustic codebooks A1..A7, teacher-forced, conditioned on the backbone hidden
  state + the codebook prefix. CSM factorization: the **semantic** codebook is the
  backbone's job, depth handles only the 7 acoustics.
- **[NEW] `depth_on_all_steps=True` (v1).** Depth is computed at every step and
  masked to frame steps in the loss. Costs ~30% of forward FLOPs (accounted for
  exactly in the isoflop budget), but keeps shapes static and the implementation
  obviously correct. A gather-to-frame-steps optimization is deferred to v2.
- **[NEW] Depth sizing rule: d_depth = d_backbone / 2, 4 layers (~7% of params).**
  From the published CSM/Moshi ratios (CSM: 1B/100M=10%, 3B/250M=8.3%, 8B/300M=
  3.75%, shrinking with scale; Moshi depth 6L/d1024 is a few %). Our 170-500M
  backbones sit below CSM-Tiny, so ~7% brackets the plausible range. **P3 ablates
  this** (dd=256/L2 ≈3% and dd=512/L6 ≈13%) to show the rule isn't load-bearing.
- **[NEW] Shared 144,644-vocab embedding table** for the backbone input-sum;
  separate small depth embeddings and per-codebook heads (mirrors CSM, which
  projects the backbone hidden and uses separate codebook embeddings).

## Hyperparameters

- **[PORTED] Model dims from width d:** `layers = round(d/(64 + 4·log2 d − 7))`,
  `heads = kv = max(1, d//128)` (MHA), `intermediate = 4d`. Verbatim from the old
  sweep constants (base_hidden_layer_ratio=64, hidden_head_ratio=128, MLP_RATIO=4).
- **[PORTED] Global batch:** pow-2 of `(budget/3)/(fwd_per_unit · 2^16 · seq)`,
  min 8, halved while lr exceeds the cap. The 2^16 anchor ties batch to budget.
- **[PORTED] LR = 0.33·√B / d, capped at 0.01.** `lr_constant=0.33` is the old
  sweep's empirical constant; the √batch/width form is muP-grounded.
- **[PORTED] beta2 = 0.98^(B/128)** (batch-scaled, arXiv:2507.07101); beta1=0.95,
  eps=1e-15; **CautiousConfig** optimizer, wd=0.1, warmup=0.1, linear schedule,
  decay=0.2, min_lr_ratio=0, adamc_weight_decay=True — all from the old sweep.
- **[PORTED] steps = (budget/3)/(fwd_per_unit · B · seq).**
- **[DEVIATION] z-loss = 1e-4.** The old *sweep* used no z-loss; the *production*
  runs used 1e-4. We apply 1e-4 (stabilizes the 144k/130k-vocab softmax). Applied
  to **both arms**, so it cannot confound the flattened-vs-hierarchical comparison;
  it only affects exact reproduction of the sweep's loss curve. One-line revert to
  0 if exact-sweep-match is wanted.
- **[NEW] L_STEPS = 1024** (Arm H window length). No old precedent; chosen so a
  window holds nearly all utterance-docs intact while keeping the attention-seq
  FLOP term small relative to the MLP.
- **[NEW] Arm H batch uses the *same* formula** with per-step FLOPs, seq=1024, and
  the **backbone** width. Yields B=32 at 3e18 (vs Arm F B=8). **Caveat for the
  writeup:** because a step packs 8 codebook tokens, the Arm H batch covers ~6-8x
  more underlying audio per gradient update than Arm F's 32,768 flat tokens. The
  batch-size choice is the least-anchored knob here; it is defensible because the
  lr auto-scales via √batch/width, B=32 is inside the old sweep's 8-256 range, and
  batch is second-order to architecture in an isoflop comparison. Alternative
  (token-matched batching → hier B≈4-8) is a one-line change; not adopted.

## Loss weighting (the 2x2 ablation)

- **[PORTED / NEW] Arm F uniform** (the published SODA recipe); **Arm H
  Moshi-weighted** (alpha=100 on text+semantic terms, 1 on acoustic; weighted-mean
  normalization `Σα·ce / Σα·w`), verified against Moshi Table 1 / Eq. 7. Both are
  the headline pair (P1). To decompose *architecture* from *weighting*, P1b runs
  Arm H uniform and P1c (time-permitting) runs Arm F Moshi-weighted — the full 2x2.
  alpha is a config field on both arms, so all four cells are runnable.

## Evaluation

- **[NEW] Held-out NLL first; headline = bits per audio-second.** Raw summed NLL
  is comparable across factorizations because both arms model the same joint over
  the same tokens (`flat == steps + 7·frames` per doc, asserted at preprocessing).
  bits/audio-second normalizes by real audio duration (12.5 Hz), so it is
  factorization-agnostic. Per-codebook and text/semantic splits are also logged.
- **[NEW] Primary set = LibriSpeech dev-clean** (`librispeech-mm-eval`,
  dev_clean_asr) — clean, standard, independent of our training-subset choices;
  secondary = the in-domain 2-per-mille YODAS/Emilia holdout.
- **[NEW] Both-window truncation:** eval docs are truncated to <=480 frames and
  <=512 text tokens so a doc fits *both* the 4096-flat and 1024-step windows —
  every reported number covers identical content in both arms. One doc per
  sequence, teacher-forced.
- **[DEFERRED] blueberry-eval (sWUGGY/sBLIMP/Salmon) + HF export** — a stretch on
  selected checkpoints after the NLL frontier exists.

## Infrastructure / process

- **[NEW] One shared thin train main** (`train_audio_lm.py`) for both arms: same
  Trainer, optimizer build, seed->key split, mixture/feistel-shuffle, mp policy,
  checkpointer. Parity is structural, not by convention — the trainer is never the
  variable between arms.
- **[NEW] Config-hash run names** (Marin-canonical `fingerprint_hash(canonical_json)`,
  8 hex, e.g. `p1-hier-moshi-765a7592`). The hash covers the experiment (model,
  data, optimizer, loss recipe, global batch/steps/seed) but NOT infra (device
  count, mesh, paths), so 1<->N GPU swaps keep the same hash and resume the same
  checkpoint. Prevents the two failure modes we hit: (a) same-name configs
  colliding — p1-hier and p1b-hier share a solver spec but get distinct hashes;
  (b) a re-run silently resuming a stale checkpoint (which once produced a bogus
  train/loss 0.0 no-op).
- **Data-parallel is exact:** the 2-GPU smoke reproduced the single-GPU losses on
  both arms (8.07 / 3.93), and global batch is device-count-independent, so
  multi-GPU is pure speedup at fixed isoflop budget.
