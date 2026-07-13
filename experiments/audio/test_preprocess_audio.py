# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for mm-pretrain doc parsing and Arm-H window packing."""

import numpy as np
import pytest

from experiments.audio.audio_vocab import (
    AUDIO_END_ID,
    AUDIO_START_ID,
    BOS_ID,
    EOS_ID,
    NUM_CODEBOOKS,
    TEXT_END_ID,
    TEXT_START_ID,
    codes_to_lm_ids,
)
from experiments.audio.preprocess_audio import (
    PAD,
    DocParseError,
    WindowPacker,
    base_id_of,
    parse_doc,
)


def synthetic_doc(frames: int, text_ids: list[int], rng: np.random.Generator) -> tuple[list[int], int]:
    """[BOS, <ts>, text..., <te>, <as>, audio..., <ae>, EOS] and its expected step count."""
    codes = rng.integers(0, 2048, size=(frames, NUM_CODEBOOKS), dtype=np.int32)
    audio = codes_to_lm_ids(codes).reshape(-1).tolist()
    flat = [BOS_ID, TEXT_START_ID, *text_ids, TEXT_END_ID, AUDIO_START_ID, *audio, AUDIO_END_ID, EOS_ID]
    steps = 6 + len(text_ids) + frames  # BOS, ts, te, as, ae, EOS + text + frames
    return flat, steps


def test_parse_doc_groups_frames_and_keeps_text_order():
    rng = np.random.default_rng(0)
    flat, expected_steps = synthetic_doc(frames=3, text_ids=[100, 200, 300], rng=rng)
    flat_arr, codes8, frames = parse_doc(flat)
    assert frames == 3
    assert codes8.shape == (expected_steps, NUM_CODEBOOKS)
    # text/special steps carry their id in slot0 and PAD elsewhere
    assert codes8[0, 0] == BOS_ID and (codes8[0, 1:] == PAD).all()
    assert codes8[2:5, 0].tolist() == [100, 200, 300]
    # frame steps carry all 8 ids; reassembling them gives back the flat audio span.
    # Step layout: [BOS, <ts>, t, t, t, <te>, <as>, f0, f1, f2, <ae>, EOS] -> frames at 7:10,
    # and the flat audio span starts after the 7 leading ids.
    frame_rows = codes8[7:10]
    assert (frame_rows != PAD).all()
    assert frame_rows.reshape(-1).tolist() == flat[7 : 7 + 24]
    # term-count identity: flat length == steps + 7*frames
    assert len(flat_arr) == len(codes8) + 7 * frames


def test_parse_doc_rejects_partial_frame():
    rng = np.random.default_rng(1)
    flat, _ = synthetic_doc(frames=2, text_ids=[7], rng=rng)
    with pytest.raises(DocParseError):
        parse_doc(flat[:-3])  # truncates mid-frame inside the audio run


def test_parse_doc_rejects_non_frame_major_audio_block():
    """Corrupt docs whose audio run is frame-length-aligned but out of codebook
    layout must skip as DocParseError, not kill the worker pool (the bare
    ValueError from lm_ids_to_codes crashed prep-yodas-3 on the v4 shards)."""
    rng = np.random.default_rng(2)
    flat, _ = synthetic_doc(frames=2, text_ids=[7], rng=rng)
    a = flat.index(AUDIO_START_ID) + 1  # first audio id
    flat[a], flat[a + 1] = flat[a + 1], flat[a]  # cross-slot swap breaks the layout
    with pytest.raises(DocParseError):
        parse_doc(flat)


@pytest.mark.parametrize(
    ("doc_id", "base"),
    [
        ("fgBC-IZVhO4_type1", "fgBC-IZVhO4"),
        ("fgBC-IZVhO4_type2", "fgBC-IZVhO4"),
        ("plain-id", "plain-id"),
        ("tricky_typeX", "tricky_typeX"),  # non-numeric suffix is not an interleave tag
    ],
)
def test_base_id_strips_only_interleave_suffix(doc_id, base):
    assert base_id_of(doc_id) == base


def test_packer_packs_multiple_docs_with_segment_ids_and_padding():
    packer = WindowPacker(length=10)
    doc_a = np.full((4, NUM_CODEBOOKS), 1, dtype=np.int32)
    doc_b = np.full((3, NUM_CODEBOOKS), 2, dtype=np.int32)
    assert list(packer.add(doc_a)) == []
    assert list(packer.add(doc_b)) == []
    (window,) = packer.flush()
    assert window["seg_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1, PAD, PAD, PAD]
    assert (window["codes8"][:4, 0] == 1).all() and (window["codes8"][4:7, 0] == 2).all()
    assert (window["codes8"][7:] == PAD).all()


def test_packer_splits_long_doc_across_windows_without_loss():
    packer = WindowPacker(length=8)
    doc = np.arange(20 * NUM_CODEBOOKS, dtype=np.int32).reshape(20, NUM_CODEBOOKS)
    windows = list(packer.add(doc)) + list(packer.flush())
    assert len(windows) == 3
    kept = np.concatenate([w["codes8"][w["seg_ids"] != PAD] for w in windows])
    assert np.array_equal(kept, doc)  # nothing lost or duplicated
    # continuation of a split doc starts a fresh segment in each new window
    assert windows[1]["seg_ids"].tolist() == [0] * 8
    assert windows[2]["seg_ids"].tolist() == [0, 0, 0, 0, PAD, PAD, PAD, PAD]


def test_packer_boundary_exact_fill_starts_next_doc_fresh():
    packer = WindowPacker(length=6)
    (window,) = packer.add(np.full((6, NUM_CODEBOOKS), 5, dtype=np.int32))
    assert (window["seg_ids"] == 0).all()
    assert list(packer.flush()) == []  # nothing buffered after an exact fill
    (window2,) = list(packer.add(np.full((6, NUM_CODEBOOKS), 6, dtype=np.int32)))
    assert (window2["seg_ids"] == 0).all()
