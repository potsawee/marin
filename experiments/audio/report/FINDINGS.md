# Findings: flattened vs. hierarchical factorization at matched compute

What the campaign taught us — the interpretation layer over the numbers,
written as input for the SODA paper revision. The per-experiment registry
(questions, fixed/varied factors, design provenance) is `EXPERIMENTS.md`, and
every number below is reproducible from `results/` (per-run NLL JSONs +
`campaign_results.csv`). The two documents go hand in hand: EXPERIMENTS.md
answers "what exactly was run and why", this one answers "what did we learn".

Three parts: **Part 1** establishes what held-out NLL can and cannot say
across architectures and loss recipes; **Part 2** is the 13-run
compute-matched comparison that selects a recipe; **Part 3** carries that
recipe to **SODA-Hier**, a 1.11B run at 39x the campaign budget — a
scaled-up P5, testing whether the pilots' conclusions survive an order of
magnitude more compute (they do — including, in §3.2, a bracketing
comparison against near-compute-optimal flattened models from the original
SODA sweep). Note the scale: at 1.17e20 FLOPs this is still an
ablation-scale run, ~110x below the 1.3e22 of SODA's published HERO run —
it extends the campaign's ladder, it is not a flagship model.

**Setup in one paragraph.** SODA (arXiv:2602.16687) flattens Mimi RVQ audio
(8 codebooks/frame: 1 semantic + 7 acoustic, 12.5 Hz) into one token stream
and trains a single decoder with uniform CE. Reviewers asked whether a
CSM/Moshi-style *hierarchical* factorization — a backbone over "steps" (one
text token or one whole frame, embeddings summed) plus a small depth
transformer predicting the 7 acoustic codebooks within each frame, trained
with Moshi's alpha=100/100/1 (text/semantic/acoustic) weighting — would be
better at matched training compute. We ran a 12-run matrix across two
isoflop budgets (1e18, 3e18 = 3x forward FLOPs), three widths (d =
512/768/896), a 2x2 of {architecture} x {loss weighting}, and a
depth-allocation ablation, plus a 13th run piloting a graded per-codebook
decay weighting on the hierarchical arm (P5, §2.2 — the recipe carried into
the Part 3 scale-up); same corpus (15.7B flat tokens ≈ 42k hours; no
Nemotron text), same tokenizer, same Qwen3 blocks, same solver-derived
hyperparameters, one seed per config. Both arms define normalized joints
over the *same* token sequences (`flat_tokens == steps + 7*frames`, asserted
at preprocessing), so held-out NLL is directly comparable — what it *means*
is Part 1. Every run was evaluated on teacher-forced NLL (LibriSpeech
dev-clean, per-token-type), ASR 0/2-shot WER (test-clean), zero-shot TTS
(seed-tts-eval English, 1088 prompts; WER via whisper-large-v3, speaker
similarity via WavLM ASV), and paired-likelihood tasks (sBLIMP, sWUGGY,
SALMon, s/tStoryCloze + text tBLIMP/tWUGGY), the speech likelihood tasks
each scored two ways: all-tokens-uniform and semantic-only (codebook 0 of
every frame).

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

Spearman correlations over the 12 matrix runs (both arms, both loss
recipes; the later P5 decay pilot is not included in these correlations):

| aligned pair | rho |
|---|---|
| bits/text-token ↔ ASR-0s WER | **+0.94** |
| semantic NLL ↔ sWUGGY (semantic-scored) | **−0.94** |
| acoustic-cb1 NLL ↔ SALMon | **−0.86** |
| acoustic-cb1 NLL ↔ TTS speaker-SIM | **−0.85** |

By contrast, the uniform aggregate (bits/audio-second) correlates in the
*wrong* direction with nearly every capability across classes (rho −0.5 to
−0.8) — its strong positives are the two acoustic metrics, SALMon (+0.86)
and TTS speaker-SIM (+0.87). Two corollaries. First, the likelihood
benchmarks are themselves NLL — contrast-pair NLL under an implicit
measure — and inherit the same sensitivity: SALMon's arm ranking flips
between uniform scoring (flat 68.6 > hier 66.7) and semantic scoring (hier
65.1 > flat 63.2). Second, the generation metrics — the only genuinely
non-likelihood evaluations — land on the same axes: ASR-WER (audio→text)
follows text NLL, TTS speaker-SIM (text→audio) follows acoustic NLL, and
TTS-WER mixes the axes (+0.54 vs text NLL across classes; within a single
family it tracks the *acoustic* component at rho ≈ +0.8 — intelligibility
of generated speech is bottlenecked by acoustic modeling once content is
adequate).

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

## Part 2. Flattened vs. Hierarchical: the campaign comparison

All runs, all metrics (paired-task scores in %, chance = 50; WER in %,
lower better; bits/a·s = uniform-measure bits per audio-second, lower
better; b/txt = bits per text token; TTS-SIM = WavLM speaker cosine, higher
better; gen% = share of the 1088 TTS prompts with a terminating generation —
where gen% < 100, TTS-WER/SIM cover only the surviving prompts and carry
survivorship bias). Machine-readable copy: `results/campaign_results.csv`.

| run | bits/a·s | b/txt | ASR-0s | ASR-2s | TTS-WER | TTS-SIM | gen% | sBLIMP | sBL-sem | sWUGGY | sWG-sem | SALMon | SAL-sem | sSC | sSC-sem | tSC | tSC-sem | tBLIMP | tWUGGY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1-flat | 562.6 | 1.93 | 37.1 | 35.6 | 31.2 | **0.332** | 76 | 49.8 | 50.4 | 55.6 | 61.8 | 68.6 | 63.2 | 50.6 | 51.0 | 58.8 | 61.3 | 62.0 | 63.5 |
| p1-hier | 658.1 | 1.11 | 19.4 | 18.1 | 29.6 | 0.230 | 100 | 50.2 | 50.4 | 57.9 | 64.1 | 66.7 | 65.1 | 51.6 | 50.4 | 61.3 | 61.7 | 64.4 | 67.3 |
| p1b-hier | 631.8 | 1.34 | 22.5 | 20.2 | **17.8** | 0.330 | 100 | 49.9 | 50.0 | 57.1 | 62.8 | 68.9 | 65.0 | 51.9 | 50.1 | 62.2 | 60.9 | 64.6 | 64.5 |
| p1c-flat | 653.4 | 1.62 | 28.5 | 26.7 | 66.7 | 0.223 | 35 | 49.5 | 50.5 | 56.3 | 63.2 | 68.6 | 65.5 | 49.9 | 50.2 | 56.4 | 61.4 | 64.6 | 64.4 |
| p2-flat-d512 | 567.7 | 2.21 | 46.9 | 49.6 | 48.1 | 0.318 | 100 | 49.7 | 50.1 | 54.9 | 61.5 | 68.5 | 62.9 | 50.8 | 51.9 | 57.6 | 57.4 | 62.5 | 63.1 |
| p2-hier-d512 | 668.1 | 1.27 | 24.2 | 23.8 | 32.2 | 0.197 | 100 | 50.3 | 50.4 | 57.0 | 63.7 | 65.9 | 64.7 | 51.7 | 51.1 | 61.9 | 62.7 | 64.2 | 65.2 |
| p2-flat-d896 | 562.2 | 1.86 | 32.2 | 31.1 | 39.7 | 0.322 | 29 | 49.6 | 50.1 | 55.8 | 62.1 | 68.9 | 63.9 | 51.1 | 50.4 | 58.7 | 60.9 | 62.4 | 62.1 |
| p2-hier-d896 | 657.2 | 1.16 | **16.1** | **15.1** | 30.0 | 0.234 | 100 | 50.2 | 50.8 | 57.7 | 64.2 | 67.2 | 64.9 | 52.1 | 51.1 | 62.2 | 62.7 | 64.7 | 65.5 |
| p3-small | 673.1 | 1.17 | 21.8 | 16.3 | 41.0 | 0.217 | 100 | 50.2 | 50.5 | **58.2** | **64.7** | 66.4 | 64.9 | **53.0** | 51.2 | 61.9 | 61.4 | 65.6 | 66.5 |
| p3-large | 656.8 | 1.25 | 19.5 | 19.2 | 28.9 | 0.227 | 100 | 50.3 | 50.7 | 57.6 | 64.4 | 66.8 | 64.9 | 52.5 | 50.7 | 60.6 | 61.6 | 63.9 | 64.9 |
| p4-flat | 586.5 | 2.64 | 43.3 | 41.8 | 47.5 | 0.301 | 63 | 49.4 | 49.6 | 54.9 | 61.2 | 68.6 | 63.5 | 50.6 | 51.2 | 56.9 | 58.6 | 61.3 | 61.5 |
| p4-hier | 674.5 | 1.31 | 20.9 | 19.3 | 46.6 | 0.194 | 100 | 50.4 | 50.2 | 57.2 | 63.2 | 66.1 | 64.1 | 52.9 | 50.6 | 61.4 | 62.1 | 63.9 | 64.8 |
| p5-decay | 650.4 | 1.16 | 18.9 | 18.3 | 23.1 | 0.290 | 100 | 49.9 | 50.5 | 57.7 | 63.8 | 67.9 | **66.5** | 51.2 | 49.4 | **63.1** | 62.1 | **65.8** | 65.8 |

(p4-* are the 1e18 runs; everything else is 3e18; p5-decay is the graded
decay-weighting pilot at p1-hier's config, §2.2. Bold = campaign best.
The CSV carries a 14th row, `soda-hier-1b` — the 1.17e20 scale-up of
Part 3; it is deliberately absent from this table, which is the
compute-matched comparison.)

### 2.1 The headline pair: p1-flat vs. p1-hier at 3e18

Per Part 1, we compare capabilities and the NLL vector, not a scalar. The
hierarchical model wins nearly every capability: it **halves ASR WER** (19.4
vs 37.1 zero-shot, 18.1 vs 35.6 two-shot), edges TTS-WER (29.6 vs 31.2 —
despite flat being scored only on its easier surviving prompts, below), and
leads sWUGGY (57.9/55.6), sWUGGY-sem (64.1/61.8), tSC (61.3/58.8), tBLIMP
(64.4/62.0) and tWUGGY (67.3/63.5). The flattened model's wins are
acoustic: SALMon under uniform scoring (68.6/66.7), TTS speaker similarity
(0.332 vs 0.230), and every acoustic NLL component (cb1–cb7 3.58–4.41 vs
4.57–4.94 nats), which also drive its 17% uniform-NLL advantage. The NLL
vector tells the same story as the capabilities: hierarchical is the better
model of semantic content (semantic 2.49/2.70, text 1.11/1.93) and the
worse model of acoustic residuals. sBLIMP and sSC are at chance for both
(consistent with SODA-v1 at this budget).

TTS generation *behavior* separates the arms beyond the scores. The
flattened model failed to produce a terminating generation on 24% of
prompts (830/1088 scored), and even its surviving transcripts are
deletion-dominated (18.2% DEL vs 10.9% SUB — truncated or incomplete
speech), where the hierarchical model terminates on all 1088 and errs by
substitution (17.8% SUB, 7.7% DEL). The pattern is arm-wide, not
pair-specific: every hierarchical run in the campaign terminates on 100% of
prompts, while four of five flattened runs drop 24–71% (gen% column
above).

### 2.2 What does loss weighting do? (2x2: p1, p1b, p1c; + decay p5)

**On the flattened arm, weighting is a large, double-edged lever.** Adding
Moshi weighting (p1-flat → p1c-flat) buys most of the hierarchical arm's
text/semantic profile — ASR 37.1 → 28.5, tBLIMP 62.0 → 64.6, sWUGGY-sem
61.8 → 63.2, SAL-sem 63.2 → 65.5 — at the cost of nearly the entire
acoustic-NLL advantage (uniform-measure 3.81 → 4.41, i.e. 91 of the 96-bit
p1 gap). And the acoustic cost is not confined to likelihoods: p1c-flat's
generation collapses — termination falls from 76% to 35% of prompts, and
TTS-WER on even the surviving third doubles (31.2 → 66.7, with SIM 0.332 →
0.223). One likelihood anomaly: tSC drops (58.8 → 56.4), the only paired
task that moves against the weighting on flat.

**On the hierarchical arm, weighting is a small lever for likelihood
capabilities — and a large one for generation.** Removing it (p1-hier →
p1b-hier) costs some ASR (19.4 → 22.5) and text NLL (1.11 → 1.34) and buys
back a little acoustic NLL (26 bits of the 96-bit gap) plus the campaign's
best uniform-scored SALMon (68.9); every other paired task moves within a
point. But TTS improves dramatically: 29.6 → **17.8** WER and 0.230 →
0.330 SIM — **p1b-hier is the campaign's best TTS model on both axes**,
reaching the flattened arm's speaker similarity while halving its WER, at
100% termination.

The asymmetry is the insight: **the depth factorization and the loss
weighting are partially redundant mechanisms for de-prioritizing
acoustics** (interaction ≈ −65 bits/audio-second). Once the architecture
delegates acoustics to a small depth module, down-weighting them in the
loss has little left to do — on flat, the loss is the only such mechanism,
so it does all the work and pays the full acoustic price. TTS is the
capability that exposes the price on both arms: synthesis is the one task
where the acoustic axis is load-bearing, and both moshi-weighted cells pay
for their tilted measure with worse, less reliable generation (2x2
TTS-WER: flat-unif 31.2, flat-moshi 66.7, hier-moshi 29.6, hier-unif
17.8).

**A third hierarchical recipe — per-codebook decay — Pareto-dominates
moshi.** The 2x2 poses moshi vs uniform as a semantics-vs-generation
trade; p5-hier (3e18/d768) tests whether a graded weighting escapes it.
Instead of flat 100/100/1 (moshi) or 1/1/1 (uniform), the acoustic
codebooks get geometrically decaying weights `w_k = 100^(1−k/7)` (cb1≈52 …
cb7=1; VoiceCraft's α=(5,1,0.5,0.1) is the same shape over EnCodec's 4
levels), concentrating the ~51% per-frame acoustic loss mass (vs uniform
87.5% / moshi 6.5%) on the perceptually dominant early codebooks and
starving the late residuals. The result matches or beats **moshi on every
semantic axis** — ASR-0s 18.9 (vs 19.4 moshi / 22.5 uniform; best of the
d768 runs — the campaign-wide best is p2-hier-d896's 16.1), tBLIMP 65.8
(campaign best), tSC 63.1 (campaign best), semantic NLL 2.485 (best),
sWUGGY 57.7 ≈ 57.9 — while recovering most of the moshi→uniform
**generation** gap: TTS-WER 29.6→23.1, TTS-SIM 0.230→0.290, SALMon
66.7→67.9, at 100% termination. Mechanistically it models cb1 as well as
uniform (acoustic-cb1 NLL 4.39 ≈ 4.40) yet keeps moshi's semantic
organization — the graded weight buys both endpoints' strengths on the
hierarchical arm, at the cost only of the highest-order residual (cb7 NLL
5.06, worst of the three). This is the recipe carried into the SODA-Hier
scale-up (Part 3).

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
window holds ~82 s and never inverts. TTS adds a reliability axis to the
width story: hierarchical is width-stable (32.2 / 29.6 / 30.0 WER, 100%
termination everywhere), while flattened termination *collapses* with
width — 100% / 76% / 29% of prompts at d512/768/896 — leaving p2-flat-d512
(48.1 WER over all 1088 prompts) the only flattened run whose TTS scores
are survivorship-free; flat SIM holds at ~0.32 at every width. (The
p2-hier-d512 1.48-epoch data caveat closed empirically — it sits exactly
on the hier width trend.)

### 2.4 Varying the depth-transformer size at fixed FLOPs (P3, hier only)

Shrinking the depth transformer from the default dd384L4 to dd256L2 costs
15 bits/audio-second (673.1 vs 658.2); growing it to dd512L6 buys only 1.4
(656.8) — the default sits at the knee of the acoustic-NLL curve.
Likelihood capabilities are nearly insensitive to depth size, and the small
variant — which pays for its lighter depth with *more backbone steps* —
posts the best hierarchical sWUGGY (58.2), sWUGGY-sem (64.7), sSC (53.0)
and tBLIMP (65.6). Generation is the exception: p3-small's TTS-WER is the
worst among the 3e18 hierarchical runs (41.0 vs 29.6 default / 28.9 large;
SIM 0.217 vs 0.230/0.227) — the acoustic-NLL cost of the under-sized depth
module, invisible to every paired benchmark, surfaces as soon as the model
must *produce* the acoustic tokens rather than score them. Everything the
likelihood suite measures lives in the backbone/semantic stream;
within-frame acoustic capacity is cheap to cut for those tasks and
impossible to buy them with — but cutting below the knee is paid for in
synthesis quality, while growing past it buys nothing anywhere (p3-large ≈
default on every metric).

### 2.5 Budget trend (1e18 → 3e18)

Tripling compute improves both arms on essentially every axis — flat ASR
43.3 → 37.1, hier 20.9 → 19.4; TTS-WER 47.5 → 31.2 (flat, with termination
63% → 76%) and 46.6 → 29.6 (hier); SIM +0.03/+0.04; sWUGGY-sem +0.6/+0.9;
tSC +1.9/−0.1 — and improves NLL by −23.9 (flat) and −16.4 (hier)
bits/audio-second, with flat's text NLL improving steeply (2.64 → 1.93
bits/token). No ordering between the arms changes at either budget, on any
metric — including the TTS split (hier better WER, flat better SIM, at
both budgets): the 1e18 anchors are a faithful, cheaper preview of the
3e18 conclusions.

### Part-2 conclusions

1. **At matched training compute, the hierarchical factorization with
   Moshi weighting is the better speech language model** on ASR, on TTS
   content accuracy and generation reliability (every hierarchical run
   terminates on 100% of TTS prompts; four of five flattened runs fail on
   24–71%), and on every semantic/lexical/text benchmark that resolves
   above chance; the flattened model is the better *acoustic* model (all
   acoustic-NLL components, uniform-scored SALMon, TTS speaker
   similarity). Which one is "right" depends on the target capability —
   and most speech-LM evaluation practice targets semantics.
2. **Loss weighting and factorization are substitutes, not complements —
   for likelihood capabilities.** Moshi weighting moves flat most of the
   way toward hier's paired-task profile (at full acoustic cost), while on
   hier it is nearly a no-op there — the architecture already made the
   trade. Generation is where the substitution breaks: acoustic
   down-weighting damages TTS on *both* arms (flat-moshi collapses to 66.7
   WER at 35% termination), and the campaign's best TTS model is
   hier+uniform (p1b: 17.8 WER, SIM 0.330 — flat-level speaker similarity
   at almost half flat's WER). The factorization supplies the semantic
   organization; leaving the loss uniform keeps the acoustics trained.
3. **Size the backbone by capabilities, not by the NLL plateau.** Width
   keeps paying on ASR/semantic tasks after NLL saturates (best model:
   p2-hier-d896); the step-based window is also a practical context
   advantage (~82 s vs ≤41 s of audio, visible in flat's 2-shot
   inversion). On flat, width additionally costs generation reliability
   (termination 100/76/29% at d512/768/896).
4. **The depth transformer should be small — but not below the knee.** Its
   size buys acoustic NLL only among likelihood metrics, and halving it
   from the dd384L4 default costs no paired-task capability; but the
   under-sized depth pays in synthesis (p3-small TTS-WER 41.0 vs 29.6),
   while growing past the knee buys nothing anywhere.
5. **Budget trends are stable**: 1e18 predicts 3e18's orderings exactly,
   supporting the use of small isoflop pilots for this design space.
6. **Inference economics favor the hierarchy**: KV length T vs 8T, and
   per-frame decode of fwd(backbone) + 7·fwd(depth) vs 8·fwd(N) — **7.0x
   fewer decode FLOPs per generated audio-second** at p1 sizes (exact
   accounting via the `audio_flops.py` accountants at each arm's training
   context: 47.4 vs 6.7 GFLOPs per audio-second; the flat model pays its
   144,644-way head on all 8 tokens per frame, the hierarchy pays the
   130,308-way head once per frame and 2,048-way heads within it).

**Known holes and caveats.** Single seed per config; TTS scores for the
four flattened runs with gen% < 100 cover only their terminating prompts
(survivorship — flat WER/SIM are optimistic there; per-run counts in
`campaign_results.csv` `tts_n_wavs`); HellaSwag/MMLU omitted (≈chance at
these budgets); correlation samples are small (n = 4–6); all capability
evals are English-only and LibriSpeech-centric for ASR.

---

## Part 3. Scaling the recipe: SODA-Hier at 39x the campaign budget

Part 2's recommendation — hierarchical factorization, graded per-codebook
decay weighting, wide backbone, small depth — was selected on 1e18/3e18
pilots. Part 3 asks whether it survives an order of magnitude more compute.
**`soda-hier-1b-branch1-53c95cb9`** is essentially **a scaled-up P5**: the
same arm and same decay recipe, at 1.11B params (d1536 backbone +
dd1152/L4 depth) on the 396k-hour `audio3` corpus for **1.17e20 FLOPs —
38.9x the campaign's 3e18 point** — evaluated on the identical battery.
Design provenance: `HERO-DECISIONS.md`.

**Keep the scale in perspective.** 1.17e20 is a large *ablation*, not a
flagship: SODA's published HERO run used 1.3e22 FLOPs, ~110x more. Part 3
therefore extends the campaign's compute ladder by an order of magnitude
and shows the recipe holds there; it does not claim a
production-competitive model.

**It is a 55%-of-plan run, and that is deliberate, not truncated.** The
plan was 95,033 steps (one epoch, 2.12e20). Cluster contention made the
remaining ~10 days unobtainable, so we exploited the affordance of the
warmup–stable–decay schedule: **any stable-phase checkpoint plus a decay
leg is a finished model for the smaller budget.** The stable trunk was
stopped at step 42,346 and a 10,000-step linear decay leg branched from it
(52,345 steps total; decay = 19.1% of the shortened run, matching the 20%
shape every P1–P5 run used, so the result stays on the campaign's schedule
line). Data seen stays under one epoch — no repeats. The trunk checkpoint
is pinned and resumable, so a larger-budget point remains available without
redoing any work.

### 3.1 The campaign's recipe scales

Same arm, same recipe, same eval suite; 38.9x the compute, 3.6x the
parameters, 9.4x the corpus (paired-task scores %, chance = 50; WER %,
lower better):

| | p5-decay (3e18, 308M) | **SODA-Hier (1.17e20, 1.11B)** |
|---|---|---|
| bits/audio-second ↓ | 650.4 | **603.5** |
| bits/text-token ↓ | 1.161 | **0.738** |
| semantic NLL ↓ | 2.485 | **2.166** |
| ASR 0-shot WER ↓ | 18.9 | **7.7** |
| ASR 2-shot WER ↓ | 18.3 | **7.2** |
| TTS-WER ↓ | 23.1 | **17.2** |
| TTS-SIM ↑ | 0.290 | **0.362** |
| sWUGGY / sWG-sem | 57.7 / 63.8 | **60.4 / 67.1** |
| tBLIMP / tWUGGY | 65.8 / 65.8 | **70.2 / 70.6** |
| SALMon / SAL-sem | 67.9 / 66.5 | **69.2 / 68.1** |
| sBLIMP / sSC | 49.9 / 51.2 | 51.8 / 50.6 |

Every metric that resolved above chance in the campaign improves, several
by large margins, and **nothing regresses**. Two results deserve emphasis:

**ASR more than halves** (18.9 → 7.7 zero-shot). The campaign's best ASR
model was p2-hier-d896 at 16.1; the scaled run is less than half that
error, on the same test-clean set with identical decode parameters.

**The acoustic axis moves with scale — against the campaign's flats.**
Part 2's honest summary was that flattened models were the better
*acoustic* models — their strongest evidence being TTS speaker similarity,
where the campaign best was p1-flat at 0.332 and every hierarchical run
sat at 0.19–0.33. SODA-Hier reaches **0.362**, above every flattened run
in the campaign, while simultaneously posting the best TTS-WER (17.2 vs
p1b-hier's 17.8). So the small-scale acoustic deficit shrinks with budget
— but the campaign flats are 3e18 models, and the harder test is against
flattened models that are *also* scaled up; §3.2 runs that comparison and
finds the acoustic and generation-quality axes (SALMon, TTS WER/SIM)
still favor compute-optimal flattened models at this scale. sBLIMP and sStoryCloze remain at chance (as at every campaign
budget), and MMLU/HellaSwag sit at chance (~0.25/0.27) — expected for a
~1B speech-first model and reported as "not above chance", not as a trend.

### 3.2 Against compute-optimal flattened SODA at scale

The question that motivated this work — and the one reviewers asked — is
flat vs. hierarchical, and Part 2 could only answer it at 1e18/3e18. The
original SODA isoflop sweep provides the missing large-scale reference:
flattened models trained near compute-optimally at their scale, evaluated
with the same blueberry harness. SODA-Hier's 1.17e20 FLOPs falls between
two of those points, giving a bracket (the * marks near-compute-optimal at
that scale; parameters bracket too — 851M / 1.11B / 1.23B):

- **Flat*-9e19** — 851M (d1408/L14), step 35,313: *less* compute than ours.
- **Flat*-1.8e20** — 1.23B (d1664/L17), step 46,636: *more* compute.

The bracketing logic: where SODA-Hier beats Flat*-1.8e20 it is **clearly
better** (it wins against 1.5x more flattened compute); where it loses to
Flat*-9e19 it is **quite likely worse** (it loses against 0.77x). Between
the brackets is a draw. Scores in %, chance = 50 except MMLU/HSW (25):

| metric | Flat*-9e19 | **SODA-Hier 1.17e20** | Flat*-1.8e20 | verdict |
|---|---|---|---|---|
| ASR 0-shot WER ↓ | 15.5 | **7.7** | 11.9 | **clearly better** |
| ASR 2-shot WER ↓ | 14.4 | **7.2** | 11.4 | **clearly better** |
| sWUGGY speech | 57.1 | **60.4** | 57.6 | **clearly better** |
| sWUGGY text | 68.3 | **70.6** | 69.6 | **clearly better** |
| sBLIMP text | 67.7 | **70.2** | 69.7 | **clearly better** |
| sBLIMP speech | 49.7 | 51.8 | 50.1 | ≈chance for all |
| MMLU / HSW-norm | 26.5 / 29.3 | 25.4 / 27.4 | 27.4 / 30.5 | ≈chance band |
| SALMon | **70.7** | 69.2 | 70.6 | **quite likely worse** |
| TTS-WER ↓ | 10.2 | 17.2 | **9.5** | **quite likely worse** |
| TTS-SIM ↑ | 0.504 | 0.362 | **0.525** | **quite likely worse** |

Held-out NLL is deliberately **not** tabulated: the Flat* models train with
uniform CE and SODA-Hier with the decay weighting, and §1.1 established
that cross-loss NLL comparisons are near-circular — each loss wins the
measure it optimizes, so any NLL row here would restate the training
objectives, not the models' quality. (For the record the pattern is
exactly as §1.1 predicts; the numbers live in each model's
`outputs/ppl/nll_results.json`.)

**The campaign's axis split survives, at scale, against compute-optimal
opponents — on both sides.** SODA-Hier is clearly better on every
semantic, lexical, text and ASR metric that resolves — beating even the
flat model with 1.5x its compute — and quite likely worse on the acoustic
and generation-quality axes: SALMon and both TTS metrics lose to even the
0.77x-compute flat. The flattened arm's home turf from Part 1 (acoustics)
extends, at compute-optimal scale, to TTS speaker similarity and TTS
intelligibility — §3.1's observation that SODA-Hier beats the *campaign*
flats on TTS does not carry over to flats that are also scaled up. Two
reading notes: SALMon here is uniform-scored only (no semantic-only
variant in the old evals; §1.2 showed its ranking is sensitive to that
choice), and the TTS gap should be read alongside the corpus caveat below
— but its size (9.5 vs 17.2 WER, 0.53 vs 0.36 SIM) is unlikely to be
data-mix alone.

Four caveats, stated so the table cannot be over-read:

- **Different training corpora.** The Flat* models were trained on the
  original SODA mix (which included ~5% Nemotron text); SODA-Hier on the
  audio-only audio3 re-pick. At pilot scale this mix shift moved flat ASR
  by ~21 points in the audio-heavy direction (EXPERIMENTS.md replication
  anchor), so part of the ASR gap may be data rather than architecture.
  The text-side wins carry the *opposite* sign — the Flat* models had
  more text data and still lose sBLIMP-text and sWUGGY-text — so the
  semantic/text conclusion is strengthened, not weakened, by the confound.
- **SODA-Hier is not compute-optimal at its budget** — width was
  CSM-anchored rather than isoflop-optimized, and the run stopped at 55%
  of its planned epoch (WSD branch). The Flat* models are near-optimal at
  theirs. This handicap makes the hierarchy's wins conservative and its
  losses ambiguous.
- **The speech likelihood tasks use N=8 codebook scoring on both sides**
  (verified in the old submission logs) — the configuration unaffected by
  the later codebook-slicing fix, so fully comparable. ASR decode
  parameters and test sets verified identical (temp 1e-4, top_p 0.8,
  seed 42, n=2,620).
- **TTS prompts are formatted to match each model's training data**, so
  each model is evaluated as intended rather than mis-prompted.

### 3.3 The decay leg is what makes a stable-phase checkpoint usable

Three checkpoints were run through the same suite: two stable-phase
(27,729 and 42,346, high LR, undecayed) and the final decayed model. The
comparison isolates something the campaign could not see, because every
campaign run was only ever evaluated after its decay:

| metric | 27,729 (stable) | 42,346 (stable) | **52,345 (decayed)** |
|---|---|---|---|
| bits/audio-second ↓ | 630.7 | 628.0 | **603.5** |
| bits/text-token ↓ | 1.129 | 1.074 | **0.738** |
| ASR 0-shot WER ↓ | 15.3 | 16.1 | **7.7** |
| TTS-WER ↓ | 20.7 | 19.0 | **17.2** |
| TTS-SIM ↑ | 0.324 | 0.334 | **0.362** |
| sWUGGY | 59.3 | 58.9 | **60.4** |
| sBLIMP | 51.4 | 50.6 | **51.8** |
| SALMon | 67.6 | 68.1 | **69.2** |

**Generation transforms; discrimination barely moves.** Over the decay leg
ASR halves (16.1 → 7.7) and bits/text-token drops 31% (1.074 → 0.738),
while sBLIMP/sWUGGY/SALMon gain only 1–2 points and StoryCloze is flat.

The 14,617 stable steps between the first two checkpoints are the control
that makes this readable: they moved ASR the *wrong way* (15.3 → 16.1) and
bits/audio-second by only −2.7, whereas the 10,000 decay steps that follow
move ASR by −8.4 points and bits/audio-second by −24.5. The effect is
therefore attributable to the annealing, not to the extra tokens.

The mechanism is consistent across metrics: decaying the LR to zero
sharpens the output distribution, which is worth little to a
contrastive/paired-likelihood task (which only needs the *ordering* of two
sequence likelihoods to be right) and worth a great deal to autoregressive
generation (where every sampled token is drawn from that distribution and
errors compound). **Practical consequence: never judge a WSD run's
generative quality from a stable-phase checkpoint** — mid-training ASR/TTS
numbers understate the finished model badly, and the gap is largest
exactly where users look.

### 3.4 Scope and caveats

- **No same-corpus flattened twin at this scale.** The Flat* baselines of
  §3.2 come from the original SODA sweep on a different data mix, so the
  1e20-scale arm comparison is bracketing evidence with confounds (listed
  in §3.2), not a controlled isoflop pair like Part 2's. (The 7.0x
  decode-FLOPs advantage of §2 conclusion 6 is analytic and
  scale-independent, so it carries over unchanged — at comparable quality
  the hierarchy generates audio for ~1/7th the FLOPs.)
- **Not a controlled isoflop point.** Versus the campaign this run changes
  budget, width *and* corpus (audio3's 396k hours vs audio2's 42k)
  simultaneously — deliberately, since the goal was the strongest model
  obtainable at this budget, not a controlled ablation. Treat §3.1 as "the
  recipe holds", not as a measured scaling exponent.
- **55.1% of the intended budget** (1.17e20 of 2.12e20), stopped for
  compute availability and closed properly with a decay leg.
- **Single epoch, single seed**, English-only, LibriSpeech-centric ASR — as
  in the campaign.
- The two stable-phase checkpoints in §3.3 are *not* a scaling ladder;
  they are mid-training snapshots of one run, and the decay confound is the
  point of that section rather than a flaw in it.
