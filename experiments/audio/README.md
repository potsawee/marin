# SODA extension: flattened vs. CSM-style hierarchical audio LMs

A compute-matched comparison of two architectures over Mimi RVQ audio tokens
(1 semantic + 7 acoustic codebooks, 2048 entries each, 12.5 Hz), trained on
utterance-level interleaved text+audio:

- **Arm F (flattened)**: the SODA architecture (arXiv:2602.16687) — one Qwen3
  decoder over the flat 144,644-vocab token stream, 4096-token windows.
- **Arm H (hierarchical)**: CSM/Moshi-style — a Qwen3 **backbone** whose
  positions are *steps* (one text/special token, or one whole audio frame as
  the sum of its 8 codebook embeddings; a unified 130,308-way head predicts
  the next step's text/special/semantic token) plus a small Qwen3 **depth**
  transformer over the 8 codebook slots (per-codebook 2048-way heads predict
  the 7 acoustic codebooks, teacher-forced, conditioned on the backbone
  hidden state). 1024-step windows.

Both arms model the same joint distribution over the same tokens: per
document `flat_tokens == steps + 7 * frames` (asserted at preprocessing), so
unweighted held-out NLL on identical documents is directly comparable.
Held-out NLL is reported as a per-token-type *vector* (text, semantic,
per-codebook acoustic) plus the uniform bits-per-audio-second aggregate;
`RESEARCH-WRITEUP.md` Part 1 covers why no single scalar can rank across
arms and loss recipes. Compute accounting follows the SODA paper's axis:
budget = 3 x forward FLOPs (`audio_flops.py`); the solver
(`isoflop_audio_target.py`) ports the old sweep's derivation rules and is
regression-pinned to its published grid.

**Status (2026-07-13).** The 13-run campaign (P1–P5) is complete: trained,
NLL-evaluated, HF-exported with JAX↔HF parity verified, and evaluated on
ASR / zero-shot TTS / the paired-likelihood suite. The per-codebook decay
weighting from the P5 pilot was selected as the release recipe, and
**SODA-Hier** (`soda-hier-1b`, 1.11B params) has been training on the 396k-hour
corpus since 2026-07-13 (~17-day epoch). W&B project: `soda-extension`.

## Reading order

| doc | contents |
|---|---|
| `EXPERIMENTS.md` | campaign registry: what each experiment asks, fixed/varied factors, result capsule |
| `RESEARCH-WRITEUP.md` | findings narrative: NLL measure-dependence (Part 1), the 12-run comparison + decay recipe (Part 2) |
| `DECISIONS.md` | campaign design rationale, every choice tagged [PORTED]/[NEW]/[DEVIATION] with provenance |
| `HERO-DECISIONS.md` | SODA-Hier release-run design log: model/recipe/data/ops decisions with measured provenance |
| `results/campaign_results.csv` | canonical numbers: all 13 runs x all metrics |
| `results/p*/` | per-run NLL eval JSONs (the full per-token-type vectors) |
| `findings/` | self-contained engineering investigations (MFU/throughput, checkpoint-resume no-op) |

`PROGRESS.local.md` and `SMOKE_LADDER.local.md` are untracked local run
trackers; docstrings that mention them refer to files that exist only in the
working checkout.

## Code layout

| file | role |
|---|---|
| `audio_vocab.py` | id-space constants; Unicode <-> codebook <-> LM-id helpers |
| `preprocess_audio.py` | parquet -> Levanter caches + holdout manifest; chunk-parallel mode (`--source/--chunk/--aggregate`) used for the HERO corpus |
| `data.py` | `AudioStepExample` + step-window dataset + source mixture (Arm H) |
| `model_hier.py` | `AudioHierConfig` / `AudioHierModel` + joint loss; per-codebook `acoustic_weights` implements the decay recipe |
| `train_audio_lm.py` | shared thin train main (one Trainer/optimizer/seed path for BOTH arms) |
| `audio_flops.py` | per-arm forward-FLOPs accountants, exact param counts, GPU-hour planning |
| `isoflop_audio_target.py` | budget x width -> RunSpec (dims, batch, lr, beta2, steps); prints the campaign table |
| `eval_audio_nll.py` | teacher-forced per-type NLL on LibriSpeech dev-clean, both-arm adapters |
| `exp_smoke.py` | P0 smoke rungs: `armf-tiny`, `overfit`, `depth0`, `decay`, `floors`, `evaldet` |
| `exp_isoflop_headline.py` | P1/P1b/P1c: the 3e18 d=768 pair + the loss-weighting 2x2 cells |
| `exp_isoflop_sweep.py` | P2: 3e18 d in {512, 896}; P4: 1e18 anchors |
| `exp_depth_ablation.py` | P3: depth-transformer allocation at 3e18 d=768 |
| `exp_hero.py` | P5 decay-weighting pilot + `soda-hier-1b` (the SODA-Hier release run) |
| `hf_export/` | HF bridge: run registry, flat exporter (stock Qwen3), hier torch port (trust_remote_code) + converter, HF-vs-JAX parity harness |
| `launchers/` | Slurm wrappers — train, NLL eval, HF export, parity, preprocessing, smokes; each header documents usage and submission conventions |
| `dispatch.py` | Fray LocalClient dispatch (bring-up era; unused by the campaign/HERO paths) |
| `test_*.py` | unit tests: solver regression-pin vs the old sweep grid, vocab round-trip, hier loss (incl. decay-weighting equivalence), preprocessing, HF-hier bit-parity |

## Data

Two corpora, same mix ratios (yodas .544 / emilia-yodas .303 / emilia .154,
no Nemotron text) and the same 2-per-mille holdout by base utterance id (the
`_type1`/`_type2` interleave twins always land on the same side):

- **Campaign corpus** (`$MARIN_PREFIX/audio2`): 15.7B flat tokens ≈ 42k
  audio-hours, 10 seeded YODAS-en shard dirs + a seeded Emilia EN file pick.
- **HERO corpus** (`$MARIN_PREFIX/audio3`): 148.03B flat tokens ≈ 396k
  audio-hours (v3+v4 seeded picks), Arm-H cache only, built as 12 chunk
  sub-sources with an aggregate manifest that asserts the realized mix
  (.546/.305/.149).

Pick manifests live under `$SODA_ROOT/data/` (outside the repo). Loss
recipes, selectable per run: uniform CE (published SODA), Moshi alpha
100/100/1 (text/semantic/acoustic), and per-codebook geometric decay
`w_k = 100^(1-k/7)` via `AudioHierConfig.acoustic_weights`.

## Workflows

Launchers submit through Stanford NLP's `nlprun`/Slurm and source
`$SODA_ROOT/env.sh`, which stays outside the repo (paths are user-specific).

```bash
# print the solver's campaign table (dims, batch, lr, steps per run)
uv run python experiments/audio/isoflop_audio_target.py

# smoke rungs: minutes on one GPU (`floors` runs on CPU)
uv run python experiments/audio/exp_smoke.py --rung decay

# train one registered run (config-hash run names; resume-safe across GPU counts)
bash experiments/audio/launchers/run_train.sh p1-hier                    # from exp_isoflop_headline.py
bash experiments/audio/launchers/run_train.sh soda-hier-1b exp_hero.py   # the HERO run

# post-hoc NLL eval; non-default hier depth geometry needs explicit flags
bash experiments/audio/launchers/run_eval.sh p2-hier-d896-<hash> hier 896
bash experiments/audio/launchers/run_eval.sh p3-hier-dd256L2-<hash> hier 768 latest --depth-hidden 256 --depth-layers 2

# HF export (CPU) + HF-vs-JAX NLL parity (1 GPU, ~5 min)
bash experiments/audio/launchers/run_export.sh hier p2-hier-d896
bash experiments/audio/launchers/run_parity.sh p2-hier-d896 16
```

Capability evals (ASR WER, zero-shot TTS, paired-likelihood tasks with
uniform and semantic-only scoring) run through the `soda-extension` branch of
blueberry-eval, outside this repo; `run_export.sh` writes the HF checkpoint
layout it consumes (`$MARIN_PREFIX/audio2-runs/<run>/hf/step-N`).
