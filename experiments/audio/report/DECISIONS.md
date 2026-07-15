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
- **[DEFERRED → DONE 2026-07-10/12] blueberry-eval + HF export.** Originally a
  stretch goal; ultimately every run (12-run matrix + the P5 pilot) was
  HF-exported with enforced JAX↔HF parity and evaluated on the full suite —
  ASR, zero-shot TTS, and the paired-likelihood tasks in uniform +
  semantic-only scoring. Workflow and results: EXPERIMENTS.md ("Capability
  results", "HF export + capability-eval bridge"), FINDINGS.md Part 2.

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
  train/loss 0.0 no-op). Full postmortem: Appendix B.
- **Data-parallel is exact:** the 2-GPU smoke reproduced the single-GPU losses on
  both arms (8.07 / 3.93), and global batch is device-count-independent, so
  multi-GPU is pure speedup at fixed isoflop budget.
- **[NEW] 1 GPU per campaign run.** The P1 throughput investigation (Appendix A)
  measured ~6% MFU, flat in batch size — more GPUs per run buy sub-linear
  speedup, so the campaign ran one config per GPU (~1.6x more
  GPU-hour-efficient than the 4-GPU launch it replaced).

---

*Appendices: long-form evidence behind the decisions above — each states its
question, method, conclusion, and the action taken.*

## Appendix A — P1 throughput: ~6% MFU, and why 1 GPU/run beats 4 GPU/run

**Date:** 2026-07-08 (during the P1 headline runs).

### Question

P1 launched on 4 GPUs each and logged **~1.0 s/step → ~17h/run**, not the ~2h I
projected. Why, and can it be fixed before committing the whole campaign?

### TL;DR

The runs are **compute-bound at ~6% MFU**, and that is largely **inherent** to
these configs on this hardware — a small model (d=768) with a 144k-vocab head on
the RTX 6000 Ada's GDDR6 bandwidth. It is **not** data-loader-bound, **not**
attention-bound, and **not** fixable by changing the batch size. The actionable
consequence is a **scheduling** change: run each campaign config on **1 GPU** and
parallelize across GPUs, rather than 4 GPUs per run.

(The earlier "~2h" estimate assumed 30% MFU — far too optimistic for small models
with a huge softmax on a workstation GPU. Corrected here.)

### What was measured

All on RTX 6000 Ada (jagupard39), the p1-flat config (d=768, 8 layers, vocab
144,644, seq 4096). Scripts: `../benches/{attn_bench,step_bench}.py`.

**1. Data loader is NOT the bottleneck.** Raw `TreeCache` random reads serve
~4.2 batch/s, the full step-window pipeline ~13.6 batch/s — both far above the
~1.0 batch/s the run consumes. (Login-node measurement; the compute node adds
contention but not a 13x gap.)

**2. Attention is NOT the bottleneck.** Micro-benchmark at the exact attention
shape (B=2, H=6, S=4096, D=128):

| impl | ms/call |
|---|---|
| materialized (current VANILLA fallback) | 2.25 |
| `jax.nn.dot_product_attention(impl="xla")` | 3.15 |
| `jax.nn.dot_product_attention(impl="cudnn")` | **0.41** |

cuDNN flash is 5.5x faster, but materialized attention is only ~2.25 ms/call →
~54 ms of a ~490 ms step (~10%). Flash would cut the step ~8%, not 5x. So
`transformer_engine`/cuDNN integration is **not** worth the dependency risk now.

**3. It's compute-bound, and batch size doesn't help.** Real model train step
(fwd+bwd via `compute_next_token_loss`) on **synthetic data** (loader removed),
1 GPU:

| per-device batch | ms/step | MFU | tok/s |
|---|---|---|---|
| 2 | 490 | 6.6% | 17k |
| 8 | 2284 | 5.6% | 14k |
| 32 | 9222 | 5.6% | 14k |

MFU is flat at ~6% and throughput flat at ~14–17k tok/s **regardless of batch**.
So the low MFU is a per-token inefficiency that does not amortize with batch —
consistent with being memory-bandwidth-bound (small d, but a 144k×768 head and
GDDR6 ~960 GB/s vs an A100's ~2 TB/s HBM). The live 4-GPU run is *worse* (~3–4%
MFU) because splitting the small global batch to 2/GPU adds all-reduce + loader
overhead on top.

### Consequence: schedule 1 GPU per run

Because MFU doesn't improve with batch and 4-GPU adds comm overhead on tiny
per-device batches, more GPUs per run buys **sub-linear** speedup:

| runs a 3e18 point as | wall/run | GPU-h/run |
|---|---|---|
| 4 GPU (measured live) | ~17 h | ~68 |
| 1 GPU (14k tok/s × 2.11B tok) | ~42 h | ~42 |

1 GPU/run is **~1.6x more GPU-hour-efficient**. For campaign throughput on N GPUs
(one config per GPU, run concurrently), 1-GPU-per-run wins decisively — e.g. 8
runs on 8 GPUs finish in ~42h at 1 GPU/run vs ~68h at 4 GPU/run (2 concurrent,
4 waves). A 2-GPU/run middle ground (~24h/run, ~48 GPU-h) trades some efficiency
for lower per-run latency.

**Actions taken:**
- Let the already-running P1 pair finish on 4 GPUs (first data point ~16h out; no
  benefit to killing correct, descending runs).
- Run the remaining campaign (P2/P1b/P4/P3, +P1c) at **1 GPU per config**,
  launching as many concurrently as GPUs allow. Runs are checkpointed +
  config-hashed, so long 1-GPU runs survive preemption and resume.
- Do **not** pursue flash-attention/transformer_engine now (≤~8% gain, real
  dependency risk). Revisit only if a kernel-level throughput push is warranted.

### Loose ends / not chased

- Whether a fused-CE block-size or an XLA flag squeezes the 144k-head matmul is
  unexplored; possible few-% wins, not a 5x.
- Hier-arm step efficiency was inferred from the live run (1.17 s/step, similar
  MFU), not separately micro-benchmarked; the depth transformer's tiny 8-long
  ops are a plausible additional drag worth a look if hier becomes the focus.

## Appendix B — Multi-GPU smoke "passed" with train/loss 0.0 — a stale-resume no-op

**Date:** 2026-07-08 (validating multi-GPU before scaling the campaign).

### Question

A 2-GPU smoke of both arms reported `RUNG ... PASS: final train loss 0.000`. A
real cross-entropy never hits exactly 0.0 — was multi-GPU corrupting training?

### Conclusion

**Not a multi-GPU bug.** The smoke used a fixed run id (`smoke-armf-tiny`) with a
persistent checkpointer. The 2-GPU run found the *previous* single-GPU run's
`step-99` checkpoint, saw it was already at the target step count, and ran **zero
training steps**:

```
Loading checkpoint from .../smoke-armf-tiny/ckpt/.../step-99
Training already complete at step 100 (target: 100). Running final hooks only.
```

`train/loss: 0.0` was an unpopulated metric (no step ran). The assertion
`loss < 9.0` passed it falsely. Data-parallel itself is correct: after wiping the
stale checkpoints, the clean 2-GPU rerun reproduced the single-GPU losses exactly
(armf-tiny 11.9→8.07, overfit 11.8→3.93), and global batch is device-count-
independent, so multi-GPU is pure (if sub-linear, see Appendix A) speedup at
fixed isoflop budget.

### Fixes (committed)

- **Config-hash run names** (`exp_isoflop_headline.experiment_signature`): the
  Marin-canonical `fingerprint_hash(canonical_json(...))` over the experiment
  (model/data/optimizer/loss recipe/global batch/steps/seed) but NOT infra
  (device count, mesh, paths). Same experiment → same hash → correct resume
  across GPU counts; different config → different hash → no stale-checkpoint
  collision. Verified: `p1-hier` and `p1b-hier` share a solver spec but get
  distinct hashes.
- **Hardened smoke** (`exp_smoke.py`): each rung wipes its store before training,
  and asserts a plausible loss floor (`0.5 < loss`) so a no-op resume fails
  loudly instead of passing.
