# P1 throughput: ~6% MFU, and why 1 GPU/run beats 4 GPU/run

**Date:** 2026-07-08 (during the P1 headline runs).

## Question

P1 launched on 4 GPUs each and logged **~1.0 s/step → ~17h/run**, not the ~2h I
projected. Why, and can it be fixed before committing the whole campaign?

## TL;DR

The runs are **compute-bound at ~6% MFU**, and that is largely **inherent** to
these configs on this hardware — a small model (d=768) with a 144k-vocab head on
the RTX 6000 Ada's GDDR6 bandwidth. It is **not** data-loader-bound, **not**
attention-bound, and **not** fixable by changing the batch size. The actionable
consequence is a **scheduling** change: run each campaign config on **1 GPU** and
parallelize across GPUs, rather than 4 GPUs per run.

(The earlier "~2h" estimate assumed 30% MFU — far too optimistic for small models
with a huge softmax on a workstation GPU. Corrected here.)

## What was measured

All on RTX 6000 Ada (jagupard39), the p1-flat config (d=768, 8 layers, vocab
144,644, seq 4096). Scripts: `soda-extension/data/{attn_bench,step_bench}.py`.

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

## Consequence: schedule 1 GPU per run

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

## Loose ends / not chased

- Whether a fused-CE block-size or an XLA flag squeezes the 144k-head matmul is
  unexplored; possible few-% wins, not a 5x.
- Hier-arm step efficiency was inferred from the live run (1.17 s/step, similar
  MFU), not separately micro-benchmarked; the depth transformer's tiny 8-long
  ops are a plausible additional drag worth a look if hier becomes the focus.
