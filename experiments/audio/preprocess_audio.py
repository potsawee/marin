# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preprocess mm-pretrain parquet into the two architecture caches.

One pass over the soda-research mm-pretrain parquet files produces, from the
SAME tokenization of each kept document:

- **Arm F cache** (flattened): one row per document, ``{"input_ids": int32[L]}``
  -- the standard Levanter text-cache layout consumed via ``TokenSeqDataset``.
- **Arm H cache** (hierarchical): pre-packed windows of ``L_STEPS`` backbone
  steps, ``{"codes8": int32[L_STEPS, 8], "seg_ids": int32[L_STEPS]}``. A step is
  one text/special token (slot0 = its id, slots1..7 = PAD) or one audio frame
  (slot0 = semantic id, slots1..7 = acoustic ids). ``seg_ids`` counts documents
  within the window; padding steps carry ``seg_ids = -1``.

Documents are kept/held-out by a deterministic hash of the BASE utterance id
(``..._type1``/``..._type2`` interleave twins share a base id and always land
on the same side of every split), so the doc set is identical for both arms.
Every document asserts the term-count identity ``len(flat) == steps + 7*frames``
-- the invariant that makes the two arms' total NLL directly comparable.

Run on a CPU node (no GPU needed), e.g.:

    uv run python experiments/audio/preprocess_audio.py \
        --output $MARIN_PREFIX/audio2 --workers 6

For large corpora the build parallelizes across Slurm jobs by (source, chunk):
each invocation builds an independent sub-source cache named
``{source}-c{i:02d}`` from every ``N``-th file of the source, e.g.

    ... preprocess_audio.py --output $MARIN_PREFIX/audio3 --arms h \
        --yodas-pick $DATA/yodas-shard-pick-v3.json \
        --emilia-manifest $DATA/emilia-en-file-pick-v3.json \
        --token-budget 131e9 --source yodas --chunk 0/6

The keep/holdout threshold is always computed over the FULL source file list,
so chunking never changes which documents are kept. Sub-source mixture weights
are set token-proportionally from the per-chunk manifests (exp_hero.py). After
all chunks finish, ``--aggregate`` combines the chunk manifests into
``{output}/manifest.json`` and asserts the realized mix.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from levanter.store.cache import SerialCacheWriter
from transformers import AutoTokenizer

from experiments.audio.audio_vocab import (
    NUM_CODEBOOKS,
    SEMANTIC_HI,
    SEMANTIC_LO,
    TOKENIZER_ID,
    lm_ids_to_codes,
)

logger = logging.getLogger(__name__)

DATA_ROOT = "/nlp/scr/potsawee/workspace/soda-extension/data"
# Seed-42 sample of 10 of the 38 YODAS-en shard dirs (see plan; recorded here so
# the corpus definition lives in code).
YODAS_SHARDS = ["en001", "en006", "en007", "en100", "en106", "en107", "en109", "en113", "en115", "en120"]
# Old SODA speech mix (yodas:emilia-yodas:emilia = 131:73:37B tokens) renormalized
# without the unavailable Nemotron text component.
MIX_WEIGHTS = {"yodas": 0.544, "emilia_yodas": 0.303, "emilia": 0.154}
# Raw parquet bytes per flattened token, calibrated per source from the first
# production run's manifest (kept_tokens / keep_rate vs bytes on disk).
BYTES_PER_TOKEN = {"yodas": 2.95, "emilia_yodas": 3.24, "emilia": 3.01}

L_STEPS = 1024
PAD = -1
HOLDOUT_PERMILLE = 2

_HASH_BUCKETS = 100_000


@dataclass(frozen=True)
class SourceSpec:
    name: str
    files: tuple[str, ...]


@dataclass
class ParsedDoc:
    base_id: str
    flat_ids: np.ndarray  # (L,) int32 -- the Arm F row
    codes8: np.ndarray  # (steps, 8) int32 -- the Arm H step rows
    frames: int


class DocParseError(ValueError):
    pass


def doc_hash_bucket(base_id: str) -> int:
    """Deterministic bucket in [0, _HASH_BUCKETS) from the base utterance id."""
    digest = hashlib.blake2b(base_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _HASH_BUCKETS


def base_id_of(doc_id: str) -> str:
    """Strip the ``_typeN`` interleave-order suffix so twins share a base id."""
    head, sep, tail = doc_id.rpartition("_type")
    return head if sep and tail.isdigit() else doc_id


def flat_ids_to_steps(flat_ids: np.ndarray) -> np.ndarray:
    """Regroup a flattened id stream into (steps, 8) rows.

    Audio runs (ids >= SEMANTIC_LO) between ``<|audio_start|>``/``<|audio_end|>``
    become one step per frame; every other id becomes a text/special step with
    slots1..7 = PAD. Raises DocParseError on malformed runs so corrupt docs are
    skipped, not silently mis-grouped.
    """
    is_audio = flat_ids >= SEMANTIC_LO
    steps: list[np.ndarray] = []
    boundaries = np.flatnonzero(np.diff(is_audio.astype(np.int8))) + 1
    for run in np.split(flat_ids, boundaries):
        if run[0] >= SEMANTIC_LO:
            if len(run) % NUM_CODEBOOKS != 0:
                raise DocParseError(f"audio run of length {len(run)} not a multiple of {NUM_CODEBOOKS}")
            frames = run.reshape(-1, NUM_CODEBOOKS)
            try:
                lm_ids_to_codes(frames)  # validates per-slot codebook blocks
            except ValueError as e:
                # e.g. "ids are not a frame-major audio block": corrupt upstream
                # doc (first seen in the v4 yodas shards) - skip it, don't kill
                # the worker pool
                raise DocParseError(str(e)) from e
            steps.append(frames)
        else:
            text = np.full((len(run), NUM_CODEBOOKS), PAD, dtype=np.int32)
            text[:, 0] = run
            steps.append(text)
    return np.concatenate(steps).astype(np.int32)


def parse_doc(flat_ids: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Flat ids -> (flat, codes8, frames), asserting the term-count identity."""
    flat = np.asarray(flat_ids, dtype=np.int32)
    codes8 = flat_ids_to_steps(flat)
    frame_mask = (codes8[:, 0] >= SEMANTIC_LO) & (codes8[:, 0] < SEMANTIC_HI)
    frames = int(frame_mask.sum())
    if len(flat) != len(codes8) + 7 * frames:
        raise DocParseError(f"term-count identity violated: {len(flat)} != {len(codes8)} + 7*{frames}")
    return flat, codes8, frames


class WindowPacker:
    """Greedy multi-document packing of step rows into fixed L_STEPS windows."""

    def __init__(self, length: int = L_STEPS):
        self.length = length
        self._codes = np.full((length, NUM_CODEBOOKS), PAD, dtype=np.int32)
        self._segs = np.full(length, PAD, dtype=np.int32)
        self._pos = 0
        self._seg = 0

    def add(self, codes8: np.ndarray) -> Iterator[dict[str, np.ndarray]]:
        """Add one document; yield each window it completes."""
        offset = 0
        while offset < len(codes8):
            take = min(self.length - self._pos, len(codes8) - offset)
            self._codes[self._pos : self._pos + take] = codes8[offset : offset + take]
            self._segs[self._pos : self._pos + take] = self._seg
            self._pos += take
            offset += take
            if self._pos == self.length:
                yield {"codes8": self._codes.copy(), "seg_ids": self._segs.copy()}
                self._codes.fill(PAD)
                self._segs.fill(PAD)
                self._pos = 0
                self._seg = 0  # every new window restarts segment numbering
        if self._pos > 0:
            self._seg += 1  # doc ended mid-window; next doc is a new segment

    def flush(self) -> Iterator[dict[str, np.ndarray]]:
        if self._pos > 0:
            yield {"codes8": self._codes.copy(), "seg_ids": self._segs.copy()}
            self._pos = 0
            self._seg = 0


_WORKER_TOKENIZER = None


def _worker_init():
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ID)


# Rows per worker task. Task granularity used to be a whole parquet file, but
# at the HERO corpus's ~94% keep-rate one 11G yodas file parses into ~30G of
# int32 arrays per worker - 6 workers OOM-killed 128G build jobs and
# memory-thrashed co-resident chunks into their time limit (16139423/25/72,
# 2026-07-12). Row-range tasks cap a worker's result at a few GB; the skipped
# prefix of each slice is re-decoded (the files are single-row-group) but
# never tokenized, which costs minutes per chunk.
_ROWS_PER_TASK = 1000


def _process_rows(task: tuple[str, int, int, int]) -> dict:
    """Parse rows [row_lo, row_hi) of one parquet into compact KEPT-doc arrays.

    The keep/holdout decision happens before tokenization (only ~keep-rate of
    rows are ever tokenized), and results ship back to the writer process as a
    handful of large arrays instead of per-doc objects.
    """
    path, row_lo, row_hi, keep_buckets = task
    doc_ids: list[str] = []
    splits: list[int] = []  # 0 = train, 1 = holdout
    frames_per_doc: list[int] = []
    flat_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    skipped = 0
    holdout_below = HOLDOUT_PERMILLE * (_HASH_BUCKETS // 1000)
    pf = pq.ParquetFile(path)
    cursor = 0
    # batch_size 64 (was 256): Emilia shards from the v4 pick carry documents up
    # to ~330k tokens; a 256-doc tokenizer batch of those spikes several GB per
    # worker and OOM-killed 8-worker/64G build jobs (2026-07-12, MaxRSS 66.5G,
    # workers respawning in a loop and imap hanging on the killed slot).
    for batch in pf.iter_batches(batch_size=64, columns=["id", "text"]):
        lo, hi = cursor, cursor + batch.num_rows
        cursor = hi
        if hi <= row_lo:
            continue  # decoded but before our slice: skip without tokenizing
        if lo >= row_hi:
            break
        rows = batch.slice(max(row_lo - lo, 0), min(hi, row_hi) - max(lo, row_lo)).to_pylist()
        kept = []
        for row in rows:
            bucket = doc_hash_bucket(base_id_of(row["id"]))
            if bucket < holdout_below:
                kept.append((row, 1))
            elif bucket < keep_buckets:
                kept.append((row, 0))
        if not kept:
            continue
        encoded = _WORKER_TOKENIZER([r["text"] for r, _ in kept], add_special_tokens=False)["input_ids"]
        for (row, split), ids in zip(kept, encoded, strict=True):
            try:
                flat, codes8, frames = parse_doc(ids)
            except DocParseError:
                skipped += 1
                continue
            doc_ids.append(row["id"])
            splits.append(split)
            frames_per_doc.append(frames)
            flat_parts.append(flat)
            step_parts.append(codes8)
    if skipped:
        logger.warning("%s[%d:%d]: skipped %d malformed docs", path, row_lo, row_hi, skipped)
    flat_off = np.zeros(len(flat_parts) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in flat_parts], out=flat_off[1:])
    steps_off = np.zeros(len(step_parts) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in step_parts], out=steps_off[1:])
    return {
        "doc_ids": doc_ids,
        "splits": np.asarray(splits, dtype=np.uint8),
        "frames": np.asarray(frames_per_doc, dtype=np.int64),
        "flat": np.concatenate(flat_parts) if flat_parts else np.zeros(0, np.int32),
        "flat_off": flat_off,
        "steps": np.concatenate(step_parts) if step_parts else np.zeros((0, NUM_CODEBOOKS), np.int32),
        "steps_off": steps_off,
    }


def source_specs(data_root: str, emilia_manifest: str, yodas_shards: list[str] | None = None) -> list[SourceSpec]:
    yodas = []
    for shard in yodas_shards or YODAS_SHARDS:
        yodas.extend(sorted(str(p) for p in Path(data_root, "yodas2-mm-pretrain", shard).glob("*.parquet")))
    with open(emilia_manifest) as fh:
        picked = json.load(fh)["files"]
    emilia_root = Path(data_root, "emilia-mm-pretrain-fix")
    ey = sorted(str(emilia_root / p) for p in picked if p.startswith("Emilia-YODAS/EN/") and (emilia_root / p).exists())
    em = sorted(str(emilia_root / p) for p in picked if p.startswith("Emilia/EN/") and (emilia_root / p).exists())
    return [
        SourceSpec("yodas", tuple(yodas)),
        SourceSpec("emilia_yodas", tuple(ey)),
        SourceSpec("emilia", tuple(em)),
    ]


def keep_buckets_for(spec: SourceSpec, total_token_budget: float) -> int:
    """Deterministic per-source keep threshold (in hash buckets) hitting the mix budget."""
    available_tokens = sum(os.path.getsize(f) for f in spec.files) / BYTES_PER_TOKEN[spec.name]
    wanted = total_token_budget * MIX_WEIGHTS[spec.name]
    keep_rate = min(1.0, wanted / max(available_tokens, 1.0))
    return round(keep_rate * _HASH_BUCKETS)


def _flush_buffers(writers, flat_buf, window_buf, split: str, min_rows: int = 0) -> None:
    """Write buffered rows for one split in large batches (TensorStore round-trips are slow)."""
    if ("arm_f", split) in writers and len(flat_buf[split]) > min_rows:
        writers[("arm_f", split)].write_batch({"input_ids": flat_buf[split]})
        flat_buf[split].clear()
    if ("arm_h", split) in writers and len(window_buf[split]) > min_rows:
        windows = window_buf[split]
        writers[("arm_h", split)].write_batch({k: [w[k] for w in windows] for k in ("codes8", "seg_ids")})
        window_buf[split].clear()


def _build_source(
    output: str, name: str, files: tuple[str, ...], keep: int, arms: tuple[str, ...], workers: int
) -> dict:
    """Build the caches for one (sub-)source; returns its stats dict."""
    exemplar_f = {"input_ids": np.zeros((0,), dtype=np.int32)}
    exemplar_h = {"codes8": np.zeros((0, NUM_CODEBOOKS), dtype=np.int32), "seg_ids": np.zeros((0,), dtype=np.int32)}
    src_stats = {
        "files": len(files),
        "keep_buckets": keep,
        "docs": {"train": 0, "holdout": 0},
        "tokens": {"train": 0, "holdout": 0},
        "frames": {"train": 0, "holdout": 0},
    }
    holdout_ids = []
    tasks = [
        (f, lo, min(lo + _ROWS_PER_TASK, nrows), keep)
        for f in files
        for nrows in (pq.ParquetFile(f).metadata.num_rows,)
        for lo in range(0, nrows, _ROWS_PER_TASK)
    ]

    packers = {split: WindowPacker() for split in ("train", "holdout")}
    with contextlib.ExitStack() as stack:
        writers = {
            (arm, split): stack.enter_context(
                SerialCacheWriter(
                    f"{output}/{arm}/{name}/{split}",
                    exemplar_f if arm == "arm_f" else exemplar_h,
                    shard_name=name,
                )
            )
            for arm in arms
            for split in ("train", "holdout")
        }
        flat_buf: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
        window_buf: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "holdout": []}

        with mp.Pool(workers, initializer=_worker_init) as pool:
            for result in pool.imap(_process_rows, tasks):
                for i, doc_id in enumerate(result["doc_ids"]):
                    split = "holdout" if result["splits"][i] else "train"
                    flat = result["flat"][result["flat_off"][i] : result["flat_off"][i + 1]]
                    codes8 = result["steps"][result["steps_off"][i] : result["steps_off"][i + 1]]
                    if "arm_f" in arms:
                        flat_buf[split].append(flat)
                    if "arm_h" in arms:
                        window_buf[split].extend(packers[split].add(codes8))
                    src_stats["docs"][split] += 1
                    src_stats["tokens"][split] += len(flat)
                    src_stats["frames"][split] += int(result["frames"][i])
                    if split == "holdout":
                        holdout_ids.append({"id": doc_id, "source": name})
                for split in ("train", "holdout"):
                    _flush_buffers(writers, flat_buf, window_buf, split, min_rows=1024)
        for split, packer in packers.items():
            if "arm_h" in arms:
                window_buf[split].extend(packer.flush())
            _flush_buffers(writers, flat_buf, window_buf, split)
    src_stats["holdout_ids"] = holdout_ids
    logger.info("source %s done: docs=%s tokens=%s", name, src_stats["docs"], src_stats["tokens"])
    return src_stats


def run(
    output: str,
    data_root: str,
    emilia_manifest: str,
    token_budget: float,
    workers: int,
    *,
    arms: tuple[str, ...] = ("arm_f", "arm_h"),
    yodas_shards: list[str] | None = None,
    source: str | None = None,
    chunk: str | None = None,
) -> dict:
    """Build all sources (legacy, one process) or one (source, chunk) sub-source.

    In chunk mode (``source`` + ``chunk="i/n"``) the keep threshold still comes
    from the FULL source file list; only the file iteration is partitioned
    (round-robin, deterministic), and the caches/manifest land under the
    sub-source name ``{source}-c{i:02d}``.
    """
    specs = source_specs(data_root, emilia_manifest, yodas_shards)
    checked = specs if source is None else [s for s in specs if s.name == source]
    for spec in checked:
        if not spec.files:
            raise FileNotFoundError(f"no parquet files found for source {spec.name!r} under {data_root}")

    if source is not None:
        (spec,) = checked
        keep = keep_buckets_for(spec, token_budget)
        name = spec.name
        files = spec.files
        if chunk is not None:
            ci, n = (int(x) for x in chunk.split("/"))
            if not 0 <= ci < n:
                raise ValueError(f"chunk index {ci} out of range for /{n}")
            name = f"{spec.name}-c{ci:02d}"
            files = spec.files[ci::n]
        src_stats = _build_source(output, name, files, keep, arms, workers)
        holdout_ids = src_stats.pop("holdout_ids")
        chunk_manifest = {
            "source": spec.name,
            "name": name,
            "chunk": chunk,
            "token_budget": token_budget,
            "l_steps": L_STEPS,
            "holdout_permille": HOLDOUT_PERMILLE,
            "train_tokens": src_stats["tokens"]["train"],
            "stats": src_stats,
        }
        for arm in arms:
            with open(f"{output}/{arm}/{name}/manifest.json", "w") as f:
                json.dump(chunk_manifest, f, indent=1)
        with open(f"{output}/holdout_ids-{name}.jsonl", "w") as f:
            for rec in holdout_ids:
                f.write(json.dumps(rec) + "\n")
        return chunk_manifest

    stats: dict = {"sources": {}, "l_steps": L_STEPS, "holdout_permille": HOLDOUT_PERMILLE}
    all_holdout_ids = []
    for spec in specs:
        keep = keep_buckets_for(spec, token_budget)
        src_stats = _build_source(output, spec.name, spec.files, keep, arms, workers)
        all_holdout_ids.extend(src_stats.pop("holdout_ids"))
        stats["sources"][spec.name] = src_stats

    Path(output).mkdir(parents=True, exist_ok=True)
    with open(f"{output}/holdout_ids.jsonl", "w") as f:
        for rec in all_holdout_ids:
            f.write(json.dumps(rec) + "\n")
    stats["token_budget"] = token_budget
    stats["yodas_shards"] = yodas_shards or YODAS_SHARDS
    with open(f"{output}/manifest.json", "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def aggregate(output: str, mix_tolerance: float = 0.01) -> dict:
    """Combine chunk manifests into {output}/manifest.json and assert the mix.

    Reads every ``{output}/arm_*/*/manifest.json`` (chunk builds), sums per
    parent source, and checks each source's realized train-token share is
    within ``mix_tolerance`` of MIX_WEIGHTS.
    """
    manifests = {}
    for path in sorted(Path(output).glob("arm_*/*/manifest.json")):
        with open(path) as f:
            m = json.load(f)
        manifests[m["name"]] = m  # arms carry identical copies; last wins
    if not manifests:
        raise FileNotFoundError(f"no chunk manifests under {output}/arm_*/")

    per_source: dict[str, dict] = {}
    for m in manifests.values():
        agg = per_source.setdefault(
            m["source"],
            {
                "chunks": 0,
                "files": 0,
                "tokens": {"train": 0, "holdout": 0},
                "frames": {"train": 0, "holdout": 0},
                "docs": {"train": 0, "holdout": 0},
            },
        )
        agg["chunks"] += 1
        agg["files"] += m["stats"]["files"]
        for split in ("train", "holdout"):
            for key in ("tokens", "frames", "docs"):
                agg[key][split] += m["stats"][key][split]

    total_train = sum(s["tokens"]["train"] for s in per_source.values())
    mix_realized = {name: s["tokens"]["train"] / total_train for name, s in per_source.items()}
    for name, share in mix_realized.items():
        if abs(share - MIX_WEIGHTS[name]) > mix_tolerance:
            raise AssertionError(
                f"{name}: realized mix {share:.3f} vs target {MIX_WEIGHTS[name]:.3f} (>±{mix_tolerance})"
            )

    combined = {
        "sources": per_source,
        "sub_sources": {name: m["train_tokens"] for name, m in sorted(manifests.items())},
        "mix_realized": mix_realized,
        "train_tokens_total": total_train,
        "l_steps": L_STEPS,
        "holdout_permille": HOLDOUT_PERMILLE,
    }
    with open(f"{output}/manifest.json", "w") as f:
        json.dump(combined, f, indent=1)
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.environ.get("MARIN_PREFIX", "") + "/audio2")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--emilia-manifest", default=f"{DATA_ROOT}/emilia-en-file-pick-v2.json")
    parser.add_argument("--token-budget", type=float, default=16e9, help="total flattened-token target across sources")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", choices=["both", "f", "h"], default="both")
    parser.add_argument("--yodas-pick", default=None, help="shard-pick json (dirs_all) overriding YODAS_SHARDS")
    parser.add_argument("--source", choices=list(MIX_WEIGHTS), default=None, help="build only this source")
    parser.add_argument("--chunk", default=None, help="i/n: build every n-th file of --source as sub-source c{i}")
    parser.add_argument("--aggregate", action="store_true", help="combine chunk manifests + assert mix; no build")
    args = parser.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate(args.output), indent=1))
        raise SystemExit(0)
    if args.chunk is not None and args.source is None:
        parser.error("--chunk requires --source")
    yodas_shards = None
    if args.yodas_pick:
        with open(args.yodas_pick) as f:
            yodas_shards = json.load(f)["dirs_all"]
    arms = {"both": ("arm_f", "arm_h"), "f": ("arm_f",), "h": ("arm_h",)}[args.arms]
    result = run(
        args.output,
        args.data_root,
        args.emilia_manifest,
        args.token_budget,
        args.workers,
        arms=arms,
        yodas_shards=yodas_shards,
        source=args.source,
        chunk=args.chunk,
    )
    print(json.dumps(result, indent=1))
