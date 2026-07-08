# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Held-out NLL evaluation, directly comparable across the two arms.

Teacher-forced, ONE DOCUMENT PER SEQUENCE (no packing), on the same doc set
for both arms: docs whose flat length exceeds the flattened window (4096) or
whose step count exceeds the hierarchical window (1024) are dropped from BOTH
arms' eval, so every reported number covers identical content. Both arms share
one metric schema:

    nll/total, nll/text, nll/special, nll/semantic, nll/acoustic_{1..7}
    bits_per_audio_second   (all frame terms, log2, per second at 12.5 Hz)
    bits_per_text_token

Primary set: LibriSpeech dev-clean (soda-research/librispeech-mm-eval,
dev_clean_asr). Usage:

    uv run python experiments/audio/eval_audio_nll.py \
        --arm flat --d 768 --checkpoint $MARIN_PREFIX/audio2-runs/<run>/checkpoints/step-XXXX \
        --output eval.json
"""

import argparse
import json
import logging
import math

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pyarrow.parquet as pq
from haliax import Axis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionMask
from levanter.models.loss import maybe_fused_next_token_loss
from transformers import AutoTokenizer

from experiments.audio.audio_vocab import (
    FRAME_RATE,
    FULL_VOCAB,
    NUM_CODEBOOKS,
    TOKENIZER_ID,
    acoustic_codebook_of,
    is_acoustic_id,
    is_semantic_id,
    is_special_id,
    is_text_id,
)
from experiments.audio.data import PAD, AudioStepExample
from experiments.audio.isoflop_audio_target import FLAT_SEQ_LEN, HIER_STEPS, flat_dims, hier_dims
from experiments.audio.model_hier import AudioHierModel
from experiments.audio.preprocess_audio import parse_doc

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PARQUET = (
    "/nlp/scr/potsawee/workspace/soda-extension/data/librispeech-mm-eval/data/dev_clean_asr-00000-of-00001.parquet"
)
EVAL_BATCH = 16


def load_eval_docs(parquet_path: str, limit: int | None = None) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Parse eval docs -> (flat, codes8, frames), sorted by id, both-window filtered."""
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pf = pq.ParquetFile(parquet_path)
    table = pf.read(columns=["file_id", "text"]).to_pylist()
    table.sort(key=lambda r: r["file_id"])
    if limit:
        table = table[:limit]
    encoded = tok([r["text"] for r in table], add_special_tokens=False)["input_ids"]
    docs, dropped = [], 0
    for ids in encoded:
        flat, codes8, frames = parse_doc(ids)
        if len(flat) > FLAT_SEQ_LEN or len(codes8) > HIER_STEPS:
            dropped += 1
            continue
        docs.append((flat, codes8, frames))
    logger.info("eval docs: %d kept, %d dropped by the both-window rule", len(docs), dropped)
    return docs


def _bucket_totals(ce: np.ndarray, weight: np.ndarray, target_ids: np.ndarray) -> dict[str, tuple[float, int]]:
    """Sum CE and counts by target-id type. ce/weight/target_ids are flat arrays."""
    out: dict[str, tuple[float, int]] = {}
    masks = {
        "text": is_text_id(target_ids),
        "special": is_special_id(target_ids),
        "semantic": is_semantic_id(target_ids),
    }
    cbs = acoustic_codebook_of(target_ids)
    for k in range(1, NUM_CODEBOOKS):
        masks[f"acoustic_{k}"] = is_acoustic_id(target_ids) & (cbs == k)
    for name, mask in masks.items():
        m = mask & (weight > 0)
        out[name] = (float(ce[m].sum()), int(m.sum()))
    return out


def eval_flat(checkpoint: str, d: int, docs) -> dict:
    cfg = flat_dims(d)
    Vocab = Axis("vocab", FULL_VOCAB)
    model = cfg.build(Vocab, key=jrandom.PRNGKey(0))
    model = load_checkpoint(model, checkpoint, subpath="model")

    @eqx.filter_jit
    def per_pos_loss(m, tokens):
        named = hax.named(tokens, ("batch", "position"))
        acts = m.activations(named, attn_mask=AttentionMask.causal(), key=None)
        loss = maybe_fused_next_token_loss(
            "position", m.Embed, m.Vocab, acts, m.get_lm_head(), named, loss_weight=None, reduction=None
        )
        return loss.array

    totals: dict[str, tuple[float, int]] = {}
    for i in range(0, len(docs), EVAL_BATCH):
        chunk = docs[i : i + EVAL_BATCH]
        tokens = np.zeros((len(chunk), FLAT_SEQ_LEN), dtype=np.int32)
        weight = np.zeros((len(chunk), FLAT_SEQ_LEN), dtype=np.float32)
        targets = np.zeros((len(chunk), FLAT_SEQ_LEN), dtype=np.int32)
        for j, (flat, _, _) in enumerate(chunk):
            tokens[j, : len(flat)] = flat
            weight[j, : len(flat) - 1] = 1.0
            targets[j, : len(flat) - 1] = flat[1:]
        ce = np.asarray(per_pos_loss(model, jnp.asarray(tokens)))
        for name, (s, n) in _bucket_totals(ce.ravel(), weight.ravel(), targets.ravel()).items():
            s0, n0 = totals.get(name, (0.0, 0))
            totals[name] = (s0 + s, n0 + n)
    return totals


def eval_hier(checkpoint: str, d: int, docs, *, depth_hidden: int | None = None, depth_layers: int = 4) -> dict:
    cfg = hier_dims(d, depth_hidden=depth_hidden, depth_layers=depth_layers)
    model = AudioHierModel.init(cfg, key=jrandom.PRNGKey(0))
    model = load_checkpoint(model, checkpoint, subpath="model")

    @eqx.filter_jit
    def parts_fn(m, codes, segs):
        ex = AudioStepExample(
            codes=hax.named(codes, ("batch", "position", "codebook")),
            seg_ids=hax.named(segs, ("batch", "position")),
        )
        return m.per_type_losses(ex)

    totals: dict[str, tuple[float, int]] = {}

    def acc(name, s, n):
        s0, n0 = totals.get(name, (0.0, 0))
        totals[name] = (s0 + float(s), n0 + int(n))

    for i in range(0, len(docs), EVAL_BATCH):
        chunk = docs[i : i + EVAL_BATCH]
        codes = np.full((len(chunk), HIER_STEPS, NUM_CODEBOOKS), PAD, dtype=np.int32)
        segs = np.full((len(chunk), HIER_STEPS), PAD, dtype=np.int32)
        for j, (_, codes8, _) in enumerate(chunk):
            codes[j, : len(codes8)] = codes8
            segs[j, : len(codes8)] = 0
        parts = jax.tree.map(np.asarray, parts_fn(model, jnp.asarray(codes), jnp.asarray(segs)))
        w = parts["w_backbone"].ravel()
        buckets = _bucket_totals(parts["ce_backbone"].ravel(), w.astype(np.float32), parts["tgt_primary"].ravel())
        for name, (s, n) in buckets.items():
            acc(name, s, n)
        wd = parts["w_depth"]
        for k in range(1, NUM_CODEBOOKS):
            ce_k = parts["ce_depth"][..., k - 1]
            acc(f"acoustic_{k}", (ce_k * wd).sum(), wd.sum())
    return totals


def summarize(totals: dict[str, tuple[float, int]], docs) -> dict:
    frames = sum(f for _, _, f in docs)
    seconds = frames / FRAME_RATE
    nll = {name: s for name, (s, _) in totals.items()}
    counts = {name: n for name, (_, n) in totals.items()}
    frame_nll = nll["semantic"] + sum(nll[f"acoustic_{k}"] for k in range(1, NUM_CODEBOOKS))
    out = {
        "nll/total": sum(nll.values()),
        "counts": counts,
        "bits_per_audio_second": frame_nll / math.log(2) / seconds,
        "bits_per_text_token": (nll["text"] / counts["text"] / math.log(2)) if counts["text"] else None,
        "docs": len(docs),
        "audio_seconds": seconds,
    }
    for name in nll:
        out[f"nll/{name}"] = nll[name]
        out[f"mean_nll/{name}"] = nll[name] / counts[name] if counts[name] else None
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("flat", "hier"), required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--depth-hidden", type=int, default=None)
    parser.add_argument("--depth-layers", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-parquet", default=DEFAULT_EVAL_PARQUET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    docs = load_eval_docs(args.eval_parquet, limit=args.limit)
    if args.arm == "flat":
        totals = eval_flat(args.checkpoint, args.d, docs)
    else:
        totals = eval_hier(args.checkpoint, args.d, docs, depth_hidden=args.depth_hidden, depth_layers=args.depth_layers)
    summary = summarize(totals, docs)
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
