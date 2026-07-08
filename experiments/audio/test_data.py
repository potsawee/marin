# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip test: Arm H cache rows -> AudioStepExample through the real cache layer."""

import numpy as np
from levanter.store.cache import SerialCacheWriter

from experiments.audio.audio_vocab import NUM_CODEBOOKS
from experiments.audio.data import AudioStepDataset, step_cache_exemplar
from experiments.audio.preprocess_audio import PAD, WindowPacker


def test_cache_roundtrip_preserves_packed_windows(tmp_path):
    rng = np.random.default_rng(0)
    docs = [
        rng.integers(0, 2048, size=(n, NUM_CODEBOOKS)).astype(np.int32)
        for n in (5, 9, 3)
    ]
    packer = WindowPacker(length=8)
    windows = [w for doc in docs for w in packer.add(doc)] + list(packer.flush())

    with SerialCacheWriter(str(tmp_path), step_cache_exemplar()) as writer:
        for w in windows:
            writer.write_batch({k: [v] for k, v in w.items()})

    ds = AudioStepDataset.load(str(tmp_path)).as_sync_dataset()
    assert len(ds) == len(windows)
    for i, expected in enumerate(windows):
        ex = ds[i]
        assert np.array_equal(np.asarray(ex.codes.array), expected["codes8"])
        assert np.array_equal(np.asarray(ex.seg_ids.array), expected["seg_ids"])
        assert ex.codes.axes == (ex.codes.resolve_axis("position"), ex.codes.resolve_axis("codebook"))
        assert ex.codes.array.shape == (8, NUM_CODEBOOKS)
    # padding markers survive the trip (the model depends on them for masking)
    last = ds[len(windows) - 1]
    assert (np.asarray(last.seg_ids.array) == PAD).any()
