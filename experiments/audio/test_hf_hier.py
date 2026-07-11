# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bit-parity of the trust_remote_code torch port against AudioHierModel.

A random-weight hier model is converted in-memory with the same state-dict
mapping the exporter uses; the torch forward's gathered log-probs over a
synthetic doc must reproduce the JAX per_type_losses terms bucket-for-bucket.
Small geometries keep it CPU-fast; the second one exercises a non-128 depth
head_dim (the dd448 quirk that once crashed rotary).
"""

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
import torch

from experiments.audio.audio_vocab import (
    AUDIO_END_ID,
    AUDIO_ID_LO,
    AUDIO_START_ID,
    BOS_ID,
    CODEBOOK_SIZE,
    TEXT_END_ID,
    TEXT_START_ID,
)
from experiments.audio.data import AudioStepExample
from experiments.audio.eval_audio_nll import _bucket_totals
from experiments.audio.hf_export.convert_hier_to_hf import soda_hier_config_from, torch_state_dict_from_jax
from experiments.audio.hf_export.modeling_soda_hier import SodaHierForCausalLM
from experiments.audio.model_hier import AudioHierConfig, AudioHierModel
from experiments.audio.preprocess_audio import parse_doc

TOLERANCE_NATS = 2e-4

GEOMETRIES = [
    pytest.param(
        AudioHierConfig(
            hidden_dim=256,
            intermediate_dim=1024,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            depth_hidden_dim=128,
            depth_intermediate_dim=512,
            depth_layers=2,
            depth_heads=1,
            depth_kv_heads=1,
            z_loss_weight=None,
        ),
        id="head128",
    ),
    pytest.param(
        AudioHierConfig(
            hidden_dim=256,
            intermediate_dim=1024,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            depth_hidden_dim=224,
            depth_intermediate_dim=896,
            depth_layers=2,
            depth_heads=2,
            depth_kv_heads=2,
            z_loss_weight=None,
        ),
        id="head112-dd448-quirk",
    ),
]


def _synthetic_doc(rng: np.random.Generator, n_frames: int = 6) -> list[int]:
    text = rng.integers(10, 20000, size=5).tolist()
    frames = []
    for _ in range(n_frames):
        codes = rng.integers(0, CODEBOOK_SIZE, size=8)
        frames.extend(AUDIO_ID_LO + cb * CODEBOOK_SIZE + int(idx) for cb, idx in enumerate(codes))
    return [BOS_ID, TEXT_START_ID, *text, TEXT_END_ID, AUDIO_START_ID, *frames, AUDIO_END_ID]


def _jax_buckets(model: AudioHierModel, codes8: np.ndarray) -> dict[str, tuple[float, int]]:
    ex = AudioStepExample(
        codes=hax.named(jnp.asarray(codes8[None]), ("batch", "position", "codebook")),
        seg_ids=hax.named(jnp.zeros((1, len(codes8)), dtype=jnp.int32), ("batch", "position")),
    )
    parts = model.per_type_losses(ex)
    totals = _bucket_totals(
        np.asarray(parts["ce_backbone"]).ravel(),
        np.asarray(parts["w_backbone"]).ravel().astype(np.float32),
        np.asarray(parts["tgt_primary"]).ravel(),
    )
    wd = np.asarray(parts["w_depth"]).ravel().astype(np.float32)
    ce_dep = np.asarray(parts["ce_depth"]).reshape(-1, ce_depth_terms(parts))
    for k in range(ce_dep.shape[-1]):
        totals[f"acoustic_{k + 1}"] = (float((ce_dep[:, k] * wd).sum()), int(wd.sum()))
    return totals


def ce_depth_terms(parts) -> int:
    return parts["ce_depth"].shape[-1]


def _torch_buckets(model: SodaHierForCausalLM, flat: np.ndarray) -> dict[str, tuple[float, int]]:
    ids = torch.tensor(flat, dtype=torch.long)[None]
    with torch.no_grad():
        logits = model(ids).logits[0]
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    ce = -logp.gather(1, tgt[:, None])[:, 0]
    return _bucket_totals(ce.numpy(), np.ones(len(ce), dtype=np.float32), tgt.numpy())


@pytest.mark.parametrize("cfg", GEOMETRIES)
def test_torch_port_matches_per_type_losses(cfg):
    jax_model = AudioHierModel.init(cfg, key=jrandom.PRNGKey(0))
    torch_model = SodaHierForCausalLM(soda_hier_config_from(cfg))
    state = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in torch_state_dict_from_jax(jax_model).items()}
    torch_model.load_state_dict(state, strict=True)
    torch_model.eval()

    rng = np.random.default_rng(7)
    flat, codes8, frames = parse_doc(_synthetic_doc(rng))
    assert frames == 6

    jx = _jax_buckets(jax_model, codes8)
    th = _torch_buckets(torch_model, flat)

    for name in sorted(set(jx) | set(th)):
        js, jn = jx.get(name, (0.0, 0))
        ts, tn = th.get(name, (0.0, 0))
        assert jn == tn, f"{name}: count mismatch jax={jn} torch={tn}"
        if jn:
            assert abs(js - ts) / jn <= TOLERANCE_NATS, f"{name}: mean delta {(js - ts) / jn:.2e}"
