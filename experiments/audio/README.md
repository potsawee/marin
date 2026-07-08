# Flattened vs CSM-style hierarchical audio LMs

A compute-matched comparison of two architectures over Mimi RVQ audio tokens
(1 semantic + 7 acoustic codebooks, 2048 entries each, 12.5 Hz), trained on the
soda-research mm-pretrain corpora (utterance-level interleaved text+audio):

- **Arm F (flattened)**: the SODA architecture — one Qwen3 decoder over the
  flat 144,644-vocab token stream, 4096-token windows.
- **Arm H (hierarchical)**: CSM/Moshi-style — a Qwen3 **backbone** whose
  positions are *steps* (one text/special token, or one whole audio frame as
  the sum of its 8 codebook embeddings; unified 130,308-way head predicts the
  next step's text/special/semantic token) plus a small Qwen3 **depth**
  transformer over the 8 codebook slots (per-codebook 2048-way heads predict
  the 7 acoustic codebooks, teacher-forced, conditioned on the backbone hidden
  state). 1024-step windows.

Both arms model the same joint distribution over the same tokens: per document
`flat_tokens == steps + 7 * frames` (asserted at preprocessing), so unweighted
summed NLL over identical held-out documents is directly comparable. The
headline metric is **bits per audio-second** on LibriSpeech dev-clean.

Compute accounting follows the SODA paper's axis: budget = 3 x forward FLOPs
(`audio_flops.py`); the solver (`isoflop_audio_target.py`) ports the old
sweep's derivation rules and is regression-pinned to its published grid.

## Files

| file | role |
|---|---|
| `audio_vocab.py` | id-space constants; Unicode <-> codebook <-> LM-id helpers |
| `preprocess_audio.py` | one-pass parquet -> both arms' Levanter caches + holdout manifest |
| `data.py` | `AudioStepExample` + step-window dataset + source mixture (Arm H) |
| `model_hier.py` | `AudioHierConfig` / `AudioHierModel` + joint loss + per-type losses |
| `train_audio_lm.py` | shared thin train main (one Trainer/optimizer/seed path for BOTH arms) |
| `dispatch.py` | Fray LocalClient dispatch (grug pattern) |
| `audio_flops.py` | per-arm forward-FLOPs accountants, exact param counts, GPU-hour planning |
| `isoflop_audio_target.py` | budget x width -> RunSpec (dims, batch, lr, beta2, steps); campaign table |
| `eval_audio_nll.py` | teacher-forced per-type NLL on LibriSpeech dev-clean, both-arm adapters |
| `exp_smoke.py` | P0 smoke rungs (see SMOKE_LADDER.local.md) |
| `exp_isoflop_headline.py` | P1/P1b/P1c: the 3e18 d=768 pair + loss-weighting cells |
| `exp_isoflop_sweep.py` | P2: 3e18 d in {512, 896}; P4: 1e18 anchors |
| `exp_depth_ablation.py` | P3: depth-size ablation at 3e18 d=768 |

## Data

Corpus definition lives in `preprocess_audio.py`: 10 seeded YODAS-en shard dirs
+ a seeded Emilia EN file pick (manifests under
`/nlp/scr/potsawee/workspace/soda-extension/data/`), old SODA mix ratios
(yodas .544 / emilia-yodas .303 / emilia .154) without the unavailable
Nemotron text. Holdout is 2 per-mille by base utterance id (the
`_type1`/`_type2` interleave twins always land on the same side). Loss
recipes: Arm F uniform (the published SODA recipe); Arm H Moshi-weighted
(alpha 100 text/semantic, 1 acoustic) — both arms support both, giving the
2x2 architecture x weighting ablation.

## Running

```bash
# solve + print the campaign
uv run python experiments/audio/isoflop_audio_target.py

# preprocessing and training run as Slurm jobs; see the launcher scripts
# (run_preprocess.sh, run_train.sh, run_smoke_gpu.sh) under
# /nlp/scr/potsawee/workspace/soda-extension/

# post-hoc eval of any checkpoint
uv run python experiments/audio/eval_audio_nll.py \
    --arm hier --d 768 --checkpoint .../checkpoints/<run>/step-NNNN --output eval.json
```
