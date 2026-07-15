# Findings: flattened vs. hierarchical factorization at matched compute

What the campaign taught us — the interpretation layer over the numbers,
written as input for the SODA paper revision. The per-experiment registry
(questions, fixed/varied factors, design provenance) is `EXPERIMENTS.md`, and
every number below is reproducible from `results/` (per-run NLL JSONs +
`campaign_results.csv`). The two documents go hand in hand: EXPERIMENTS.md
answers "what exactly was run and why", this one answers "what did we learn".

**Setup in one paragraph.** SODA (arXiv:2602.16687) flattens Mimi RVQ audio
(8 codebooks/frame: 1 semantic + 7 acoustic, 12.5 Hz) into one token stream
and trains a single decoder with uniform CE. Reviewers asked whether a
CSM/Moshi-style *hierarchical* factorization — a backbone over "steps" (one
text token or one whole frame, embeddings summed) plus a small depth
transformer predicting the 7 acoustic codebooks within each frame, trained
with Moshi's alpha=100/100/1 (text/semantic/acoustic) weighting — would be
better at matched training compute. We ran a 12-run matrix across two
isoflop budgets (1e18, 3e18 = 3x forward FLOPs), three widths
(d = 512/768/896), a 2x2 of {architecture} x {loss weighting}, and a
depth-allocation ablation, plus a 13th run piloting a graded per-codebook
decay weighting on the hierarchical arm (P5, §2.2 — the SODA-Hier release
recipe); same corpus (15.7B flat tokens ≈ 42k hours; no Nemotron text), same
tokenizer, same Qwen3 blocks, same solver-derived hyperparameters, one seed
per config. Both arms define normalized joints over the *same* token
sequences (`flat_tokens == steps + 7*frames`, asserted at preprocessing), so
held-out NLL is directly comparable — what it *means* is Part 1. Every run
was evaluated on teacher-forced NLL (LibriSpeech dev-clean, per-token-type),
ASR 0/2-shot WER (test-clean), zero-shot TTS (seed-tts-eval English, 1088
prompts; WER via whisper-large-v3, speaker similarity via WavLM ASV), and
paired-likelihood tasks (sBLIMP, sWUGGY, SALMon, s/tStoryCloze + text
tBLIMP/tWUGGY), the speech likelihood tasks each scored two ways:
all-tokens-uniform and semantic-only (codebook 0 of every frame).

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
decay-weighting pilot at p1-hier's config, §2.2. Bold = campaign best.)

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
5.06, worst of the three). This is the recipe selected for the SODA-Hier
release run (Part 3).

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
