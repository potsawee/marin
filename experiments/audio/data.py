# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data plumbing for the hierarchical (Arm H) audio LM.

The Arm H cache (written by ``preprocess_audio.py``) stores pre-packed windows
of ``L_STEPS`` backbone steps. This module loads those windows as
:class:`AudioStepExample` batches and mixes sources with the same
block-deterministic mixture + feistel shuffle the flattened arm uses.
"""

import functools

import equinox as eqx
import haliax as hax
import jax
import numpy as np
from haliax import NamedArray
from levanter.data.dataset import AsyncDataset, MappedAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.store.cache import TreeCache

from experiments.audio.audio_vocab import NUM_CODEBOOKS

# Mirrors preprocess_audio.PAD without importing the (pyarrow-heavy) module.
PAD = -1


class AudioStepExample(eqx.Module):
    """One window of backbone steps.

    ``codes``: int32 NamedArray {position, codebook} of LM ids. Slot 0 is the
    step's primary id (text/special/semantic); slots 1..7 are the frame's
    acoustic ids or PAD on text/special steps. Padding steps are PAD in every
    slot. ``seg_ids``: int32 NamedArray {position}, document segment within the
    window; PAD on padding steps. The model derives every mask (frame steps,
    loss weights, attention segments) from these two arrays. The DataLoader
    stacks examples, prepending a batch axis to both.
    """

    codes: NamedArray
    seg_ids: NamedArray


def step_cache_exemplar() -> dict[str, np.ndarray]:
    """Row exemplar of the Arm H cache (shared with the preprocessing writer)."""
    return {
        "codes8": np.zeros((0, NUM_CODEBOOKS), dtype=np.int32),
        "seg_ids": np.zeros((0,), dtype=np.int32),
    }


def _single_cpu_sharding() -> jax.sharding.SingleDeviceSharding:
    return jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])


class AudioStepDataset(MappedAsyncDataset[dict, AudioStepExample]):
    """Maps Arm H cache rows to :class:`AudioStepExample` (one row = one window)."""

    def __init__(self, cache: AsyncDataset[dict]):
        self.cache = cache
        sharding = _single_cpu_sharding()

        @functools.partial(eqx.filter_jit)
        def _create(codes8: jax.Array, seg_ids: jax.Array) -> AudioStepExample:
            example = AudioStepExample(
                codes=hax.named(codes8, ("position", "codebook")),
                seg_ids=hax.named(seg_ids, ("position",)),
            )
            return jax.lax.with_sharding_constraint(example, sharding)

        def _map(row: dict) -> AudioStepExample:
            # pyrefly: ignore[bad-return]  # eqx.filter_jit wrapper types the call as returning Unknown
            return _create(row["codes8"], row["seg_ids"])

        super().__init__(cache, _map)

    @classmethod
    def load(cls, cache_dir: str) -> "AudioStepDataset":
        return cls(TreeCache.load(cache_dir, step_cache_exemplar()))

    async def async_len(self) -> int:
        return await self.cache.async_len()


def build_step_mixture(
    cache_dirs: dict[str, str],
    weights: dict[str, float],
    *,
    key: jax.Array,
    block_size: int = 2048,
) -> MixtureDataset[AudioStepExample]:
    """Feistel-shuffled block-deterministic mixture over per-source Arm H caches."""
    shuffle_key, mix_key = jax.random.split(key)
    datasets = {
        name: AudioStepDataset.load(path).shuffle(jax.random.fold_in(shuffle_key, i))
        for i, (name, path) in enumerate(sorted(cache_dirs.items()))
    }
    return MixtureDataset(datasets, weights, block_size, key=mix_key)
