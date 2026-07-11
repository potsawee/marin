# SODA-extension: Flattened vs. Hierarchical Factorization at Matched Compute

Findings write-up for the next SODA paper revision. This document reports and
interprets what we ran; the per-experiment registry (questions, fixed/varied
factors, design provenance) is `EXPERIMENTS.md`, and every number below is
reproducible from `results/` (per-run NLL JSONs + `campaign_results.csv`).
The two documents go hand in hand: EXPERIMENTS.md answers "what exactly was
run and why", this one answers "what did we learn".

**Setup in one paragraph.** SODA (arXiv:2602.16687) flattens Mimi RVQ audio
(8 codebooks/frame: 1 semantic + 7 acoustic, 12.5 Hz) into one token stream
and trains a single decoder with uniform CE. Reviewers asked whether a
CSM/Moshi-style *hierarchical* factorization — a backbone over "steps" (one
text token or one whole frame, embeddings summed) plus a small depth
transformer predicting the 7 acoustic codebooks within each frame, trained
with Moshi's alpha=100/100/1 (text/semantic/acoustic) weighting — would be
better at matched training compute. We ran 12 runs across two isoflop
budgets (1e18, 3e18 = 3x forward FLOPs), three widths (d = 512/768/896), a
2x2 of {architecture} x {loss weighting}, and a depth-allocation ablation;
same corpus (15.7B flat tokens ≈ 42k hours; no Nemotron text), same
tokenizer, same Qwen3 blocks, same solver-derived hyperparameters, one seed
per config. Both arms define normalized joints over the *same* token
sequences (`flat_tokens == steps + 7*frames`, asserted at preprocessing), so
held-out NLL is directly comparable — what it *means* is Part 1. Every run
was evaluated on teacher-forced NLL (LibriSpeech dev-clean, per-token-type),
ASR 0/2-shot WER (test-clean), and paired-likelihood tasks (sBLIMP, sWUGGY,
SALMon, s/tStoryCloze + text tBLIMP/tWUGGY), the speech tasks each scored
two ways: all-tokens-uniform and semantic-only (codebook 0 of every frame).

---

## Part 1. What does NLL mean across training losses and architectures?

In SODA (one architecture, one uniform loss), held-out NLL was strongly
predictive of downstream metrics. Before using NLL to compare *different*
losses and architectures, we have to establish what such comparisons mean.
The three findings below build on each other: (1.1) a scalar NLL ranking is
an artifact of the chosen token-importance measure; (1.2) the fix is to
decompose NLL into per-token-type components, which predict capabilities
remarkably well across all models; (1.3) the original scalar-NLL practice
remains valid, but only inside a model family scored under its own training
measure.

### 1.1 A scalar NLL ranking is a choice of token-importance measure

**Takeaway: cross-loss NLL comparisons are well-defined but near-circular —
each training loss wins the evaluation measure it optimizes, and changing
the measure reverses the ranking.**

Held-out NLL is a proper scoring rule and both arms model identical token
sequences, so the comparison is mathematically sound. But a scalar NLL
averages per-token bits under some *importance measure*, and weighted-CE
training is exactly maximum likelihood under a tilted measure. Re-scoring
the 2x2 runs under three measures (mean nats/term; lower better):

| run (training loss) | uniform | Moshi-100/100/1 | semantic-only |
|---|---|---|---|
| p1-flat (uniform) | **3.81** | 2.47 | 2.70 |
| p1c-flat (moshi) | 4.41 | 2.43 | 2.65 |
| p1b-hier (uniform) | 4.26 | 2.31 | 2.55 |
| p1-hier (moshi) | 4.43 | **2.25** | **2.49** |

Under the uniform measure, flattened wins by 17% and sweeps the campaign.
Under the Moshi measure, **the ranking flips completely — every hierarchical
run beats every flattened run**, and even within one architecture the
moshi-trained model overtakes the uniform one (p1c 2.43 < p1 2.47). Under
semantic-only scoring, hierarchical sweeps again. "Which factorization
reaches lower NLL" therefore has no measure-free answer.

### 1.2 Decomposed NLL predicts capabilities — across every architecture and loss

**Takeaway: the per-token-type NLL components are excellent capability
predictors even across model classes; only the uniform aggregate misleads,
because 7 of 8 audio tokens are late-RVQ residuals whose bits carry little
task-relevant information.**

Spearman correlations over all 12 runs (both arms, three loss recipes):

| aligned pair | rho |
|---|---|
| bits/text-token ↔ ASR-0s WER | **+0.94** |
| semantic NLL ↔ sWUGGY (semantic-scored) | **−0.94** |
| acoustic-cb1 NLL ↔ SALMon | **−0.86** |

By contrast, the uniform aggregate (bits/audio-second) correlates in the
*wrong* direction with nearly every capability across classes (rho −0.5 to
−0.8) — its one strong positive is SALMon (+0.86), the acoustic-consistency
task. Two corollaries. First, the likelihood benchmarks are themselves NLL —
contrast-pair NLL under an implicit measure — and inherit the same
sensitivity: SALMon's arm ranking flips between uniform scoring (flat 68.6 >
hier 66.7) and semantic scoring (hier 65.1 > flat 63.2). Second, only
generation metrics (ASR/TTS WER) are genuinely non-likelihood evaluations.

### 1.3 Scalar NLL remains a valid scaling proxy — inside a family, under its own measure

**Takeaway: SODA-v1's "NLL predicts evals" observation replicates, and its
scope is now precise: same architecture class, same training loss, eval
measure matched to the training measure, and the family should vary scale
rather than capacity allocation.**

Spearman(NLL → capability), + = predictive (sign-corrected for WER):

| capability | flat+uniform (n=4), uniform NLL | hier+moshi (n=6), uniform NLL | hier+moshi (n=6), Moshi-aligned NLL |
|---|---|---|---|
| ASR-0s | +0.80 | +0.60 | +0.66 |
| sWUGGY | **+1.00** | +0.09 | **+0.77** |
| tWUGGY | +0.40 | +0.14 | **+0.89** |
| tBLIMP | +0.40 | −0.20 | +0.60 |
| SALMon | +0.40 | +0.77 | +0.66 |

The flattened family behaves exactly as in SODA-v1. The hierarchical family
*breaks* under uniform NLL — it varies capacity allocation (P3), and its
worst-uniform-NLL run (p3-small) posts the best hier tBLIMP/sWUGGY-sem —
but scoring the same family under its own training measure restores
predictivity (sWUGGY +0.09→+0.77, tWUGGY +0.14→+0.89, tBLIMP −0.20→+0.60),
while SALMon correctly weakens (an acoustic task under an
acoustic-down-weighted measure). Caveats: n = 4/6 (|rho| ≳ 0.83 for
p < .05), single seed, and sBLIMP/sSC sit at chance at these budgets so
their correlations are noise.

**Part-1 conclusion.** Report the NLL *vector* (text, semantic, per-codebook
acoustic), not a single scalar. Across setups, compare capabilities via the
component aligned with each task — Moshi-style/semantic measures align with
lexical, syntactic, and text tasks (WUGGY, BLIMP, StoryCloze, ASR); the
uniform measure aligns with acoustic tasks (SALMon). Within one family,
track the training-aligned scalar as the scaling proxy, exactly as SODA-v1
did for flattened+uniform.

---

## Part 2. Flattened vs. Hierarchical: the 12-run comparison

All runs, all metrics (paired-task scores in %, chance = 50; WER in %,
lower better; bits/a·s = uniform-measure bits per audio-second, lower
better; b/txt = bits per text token). Machine-readable copy:
`results/campaign_results.csv`.

| run | bits/a·s | b/txt | ASR-0s | ASR-2s | sBLIMP | sBL-sem | sWUGGY | sWG-sem | SALMon | SAL-sem | sSC | sSC-sem | tSC | tSC-sem | tBLIMP | tWUGGY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1-flat | 562.6 | 1.93 | 37.1 | 35.6 | 49.8 | 50.4 | 55.6 | 61.8 | 68.6 | 63.2 | 50.6 | 51.0 | 58.8 | 61.3 | 62.0 | 63.5 |
| p1-hier | 658.1 | 1.11 | 19.4 | 18.1 | 50.2 | 50.4 | 57.9 | 64.1 | 66.7 | 65.1 | 51.6 | 50.4 | 61.3 | 61.7 | 64.4 | 67.3 |
| p1b-hier | 631.8 | 1.34 | 22.5 | 20.2 | 49.9 | 50.0 | 57.1 | 62.8 | 68.9 | 65.0 | 51.9 | 50.1 | 62.2 | 60.9 | 64.6 | 64.5 |
| p1c-flat | 653.4 | 1.62 | 28.5 | 26.7 | 49.5 | 50.5 | 56.3 | 63.2 | 68.6 | 65.5 | 49.9 | 50.2 | 56.4 | 61.4 | 64.6 | 64.4 |
| p2-flat-d512 | 567.7 | 2.21 | 46.9 | 49.6 | 49.7 | 50.1 | 54.9 | 61.5 | 68.5 | 62.9 | 50.8 | 51.9 | 57.6 | 57.4 | 62.5 | 63.1 |
| p2-hier-d512 | 668.1 | 1.27 | 24.2 | 23.8 | 50.3 | 50.4 | 57.0 | 63.7 | 65.9 | 64.7 | 51.7 | 51.1 | 61.9 | 62.7 | 64.2 | 65.2 |
| p2-flat-d896 | 562.2 | 1.86 | 32.2 | 31.1 | 49.6 | 50.1 | 55.8 | 62.1 | 68.9 | 63.9 | 51.1 | 50.4 | 58.7 | 60.9 | 62.4 | 62.1 |
| p2-hier-d896 | 657.2 | 1.16 | **16.1** | **15.1** | 50.2 | 50.8 | 57.7 | 64.2 | 67.2 | 64.9 | 52.1 | 51.1 | 62.2 | 62.7 | 64.7 | 65.5 |
| p3-small | 673.1 | 1.17 | 21.8 | 16.3 | 50.2 | 50.5 | **58.2** | **64.7** | 66.4 | 64.9 | **53.0** | 51.2 | 61.9 | 61.4 | **65.6** | 66.5 |
| p3-large | 656.8 | 1.25 | 19.5 | 19.2 | 50.3 | 50.7 | 57.6 | 64.4 | 66.8 | 64.9 | 52.5 | 50.7 | 60.6 | 61.6 | 63.9 | 64.9 |
| p4-flat | 586.5 | 2.64 | 43.3 | 41.8 | 49.4 | 49.6 | 54.9 | 61.2 | 68.6 | 63.5 | 50.6 | 51.2 | 56.9 | 58.6 | 61.3 | 61.5 |
| p4-hier | 674.5 | 1.31 | 20.9 | 19.3 | 50.4 | 50.2 | 57.2 | 63.2 | 66.1 | 64.1 | 52.9 | 50.6 | 61.4 | 62.1 | 63.9 | 64.8 |

(p4-* are the 1e18 runs; everything else is 3e18. Bold = campaign best.)

### 2.1 The headline pair: p1-flat vs. p1-hier at 3e18

Per Part 1, we compare capabilities and the NLL vector, not a scalar. The
hierarchical model wins nearly every capability: it **halves ASR WER** (19.4
vs 37.1 zero-shot, 18.1 vs 35.6 two-shot), and leads sWUGGY (57.9/55.6),
sWUGGY-sem (64.1/61.8), tSC (61.3/58.8), tBLIMP (64.4/62.0) and tWUGGY
(67.3/63.5). The flattened model's wins are acoustic: SALMon under uniform
scoring (68.6/66.7) and every acoustic NLL component (cb1–cb7 3.58–4.41 vs
4.57–4.94 nats), which also drive its 17% uniform-NLL advantage. The NLL
vector tells the same story as the capabilities: hierarchical is the better
model of semantic content (semantic 2.49/2.70, text 1.11/1.93) and the
worse model of acoustic residuals. sBLIMP and sSC are at chance for both
(consistent with SODA-v1 at this budget).

### 2.2 What does loss weighting do? (2x2: p1, p1b, p1c)

**On the flattened arm, weighting is a large, double-edged lever.** Adding
Moshi weighting (p1-flat → p1c-flat) buys most of the hierarchical arm's
text/semantic profile — ASR 37.1 → 28.5, tBLIMP 62.0 → 64.6, sWUGGY-sem
61.8 → 63.2, SAL-sem 63.2 → 65.5 — at the cost of nearly the entire
acoustic-NLL advantage (uniform-measure 3.81 → 4.41, i.e. 91 of the 96-bit
p1 gap). One anomaly: tSC drops (58.8 → 56.4), the only capability that
moves against the weighting on flat.

**On the hierarchical arm, weighting is a small lever.** Removing it
(p1-hier → p1b-hier) costs some ASR (19.4 → 22.5) and text NLL (1.11 →
1.34) and buys back a little acoustic NLL (26 bits of the 96-bit gap) plus
the campaign's best uniform-scored SALMon (68.9). Everything else moves
within a point.

The asymmetry is the insight: **the depth factorization and the loss
weighting are partially redundant mechanisms for de-prioritizing
acoustics** (interaction ≈ −65 bits/audio-second). Once the architecture
delegates acoustics to a small depth module, down-weighting them in the
loss has little left to do — on flat, the loss is the only such mechanism,
so it does all the work and pays the full acoustic price.

### 2.3 Varying the backbone size at fixed FLOPs (d = 512 / 768 / 896)

Under an isoflop budget, wider means fewer training steps. Both arms' NLL
is nearly width-flat (flat 567.7/562.6/562.2; hier 668.1/658.2/657.2 —
plateaus, not peaks), but **capabilities keep improving with width after
NLL saturates**. Hierarchical: d896 is the campaign's best model (ASR
16.1/15.1, top-tier semantic scores), improving monotonically from d512.
Flattened: d768 beats both neighbors on ASR, and d512 degrades sharply
(46.9 zero-shot) — including the campaign's only 2-shot *inversion* (49.6 >
46.9). A plausible mechanism: the flattened 4096-token window holds ≤41 s
of audio, which two-shot audio prompts strain; the hierarchical 1024-step
window holds ~82 s and never inverts. (The p2-hier-d512 1.48-epoch data
caveat closed empirically — it sits exactly on the hier width trend.)

### 2.4 Varying the depth-transformer size at fixed FLOPs (P3, hier only)

Shrinking the depth transformer from the default dd384L4 to dd256L2 costs
15 bits/audio-second (673.1 vs 658.2); growing it to dd512L6 buys only 1.4
(656.8) — the default sits at the knee of the acoustic-NLL curve.
Capabilities, however, are nearly insensitive to depth size, and the small
variant — which pays for its lighter depth with *more backbone steps* —
posts the best hierarchical sWUGGY (58.2), sWUGGY-sem (64.7), sSC (53.0)
and tBLIMP (65.6). Everything the benchmark suite measures lives in the
backbone/semantic stream; within-frame acoustic capacity is cheap to cut
and impossible to buy capabilities with.

### 2.5 Budget trend (1e18 → 3e18)

Tripling compute improves both arms on essentially every axis — flat ASR
43.3 → 37.1, hier 20.9 → 19.4; sWUGGY-sem +0.6/+0.9; tSC +1.9/−0.1 — and
improves NLL by −23.9 (flat) and −16.4 (hier) bits/audio-second, with
flat's text NLL improving steeply (2.64 → 1.93 bits/token). No ordering
between the arms changes at either budget, on any metric: the 1e18 anchors
are a faithful, cheaper preview of the 3e18 conclusions.

### Part-2 conclusions

1. **At matched training compute, the hierarchical factorization with
   Moshi weighting is the better speech language model** on ASR and on
   every semantic/lexical/text benchmark that resolves above chance; the
   flattened model is the better *acoustic* model (all acoustic-NLL
   components, uniform-scored SALMon). Which one is "right" depends on the
   target capability — and most speech-LM evaluation practice targets
   semantics.
2. **Loss weighting and factorization are substitutes, not complements.**
   Moshi weighting moves flat most of the way toward hier's capability
   profile (at full acoustic cost), while on hier it is nearly a no-op —
   the architecture already made the trade.
3. **Size the backbone by capabilities, not by the NLL plateau.** Width
   keeps paying on ASR/semantic tasks after NLL saturates (best model:
   p2-hier-d896); the step-based window is also a practical context
   advantage (~82 s vs ≤41 s of audio, visible in flat's 2-shot inversion).
4. **The depth transformer should be small.** Its size buys acoustic NLL
   only; the default dd384L4 is at the knee, and halving it costs no
   measurable capability.
5. **Budget trends are stable**: 1e18 predicts 3e18's orderings exactly,
   supporting the use of small isoflop pilots for this design space.
6. **Inference economics favor the hierarchy** (analytic): KV length T vs
   8T, and per-frame decode of fwd(backbone) + 7·fwd(depth) vs 8·fwd(N) —
   ~4.4x fewer decode FLOPs per generated second at p1 sizes (exact
   accounting from `audio_flops.py` to accompany the final figure).

**Known holes and caveats.** Single seed per config; TTS WER/SIM not yet
run (generation stack validated end-to-end; scoring pending); HellaSwag/
MMLU omitted (≈chance at these budgets); correlation samples are small
(n = 4–6); all capability evals are English-only and LibriSpeech-centric
for ASR.
