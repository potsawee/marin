# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shape-level construct checks for the hier campaign configs.

jax.eval_shape builds the model and traces the joint loss abstractly (no
parameters allocated, no GPU), so config-geometry bugs surface in tests
instead of at job startup. Regression: dd448 once resolved to 3 heads ->
odd head_dim 149, which crashed rotary when p2-hier-d896 launched.
"""

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.audio.audio_vocab import AUDIO_ID_LO, CODEBOOK_SIZE, NUM_CODEBOOKS, SEMANTIC_LO
from experiments.audio.data import AudioStepExample
from experiments.audio.isoflop_audio_target import HIER_STEPS, attn_heads, hier_dims
from experiments.audio.model_hier import AudioHierConfig, AudioHierModel

CAMPAIGN_HIER_GEOMETRIES = [
    pytest.param(512, 256, 4, id="p2-hier-d512"),
    pytest.param(768, 384, 4, id="p1-hier-d768"),
    pytest.param(896, 448, 4, id="p2-hier-d896"),
    pytest.param(768, 256, 2, id="p3-small"),
    pytest.param(768, 512, 6, id="p3-large"),
]


def test_attn_heads_contract():
    assert attn_heads(512) == 4
    assert attn_heads(768) == 6
    assert attn_heads(896) == 7
    assert attn_heads(448) == 4  # floor division gave 3 -> head_dim 149 (the bug)
    with pytest.raises(ValueError, match="head_dim"):
        attn_heads(450)  # no even integral head_dim near the 128 convention


@pytest.mark.parametrize("d,depth_hidden,depth_layers", CAMPAIGN_HIER_GEOMETRIES)
def test_hier_campaign_config_traces(d, depth_hidden, depth_layers):
    cfg = hier_dims(d, depth_hidden=depth_hidden, depth_layers=depth_layers)
    ex = AudioStepExample(
        codes=hax.zeros((hax.Axis("position", HIER_STEPS), hax.Axis("codebook", 8)), dtype=jnp.int32),
        seg_ids=hax.zeros((hax.Axis("position", HIER_STEPS),), dtype=jnp.int32),
    )

    def build_and_loss(ex):
        model = AudioHierModel.init(cfg, key=jax.random.PRNGKey(0))
        return model.compute_joint_loss(ex)

    loss = jax.eval_shape(build_and_loss, ex)
    assert loss.shape == ()


_TINY = AudioHierConfig(
    max_steps=16,
    hidden_dim=64,
    intermediate_dim=256,
    num_layers=2,
    num_heads=2,
    num_kv_heads=2,
    depth_hidden_dim=32,
    depth_intermediate_dim=128,
    depth_layers=2,
    depth_heads=2,
    depth_kv_heads=2,
)

DECAY100 = tuple(100.0 ** (1 - k / 7) for k in range(1, NUM_CODEBOOKS))


def _toy_example() -> AudioStepExample:
    """8 text steps then 8 audio frames, one segment (positions all valid but the last)."""
    Pos, Cb = hax.Axis("position", 16), hax.Axis("codebook", NUM_CODEBOOKS)
    codes = np.full((16, NUM_CODEBOOKS), -1, dtype=np.int32)
    codes[:8, 0] = np.arange(100, 108)  # text ids
    codes[8:, 0] = SEMANTIC_LO + np.arange(8)  # semantic ids
    for k in range(1, NUM_CODEBOOKS):
        codes[8:, k] = AUDIO_ID_LO + k * CODEBOOK_SIZE + np.arange(8)
    return AudioStepExample(
        codes=hax.named(jnp.asarray(codes), (Pos, Cb)),
        seg_ids=hax.ones((Pos,), dtype=jnp.int32),
    )


def test_acoustic_weights_uniform_matches_scalar_alpha():
    ex = _toy_example()
    model = AudioHierModel.init(_TINY, key=jax.random.PRNGKey(0))
    base = model.compute_joint_loss(ex)
    uniform = dataclasses.replace(_TINY, acoustic_weights=(_TINY.alpha_acoustic,) * 7)
    model_u = dataclasses.replace(model, config=uniform)
    assert jnp.allclose(base, model_u.compute_joint_loss(ex), rtol=1e-6)


def test_acoustic_weights_decay_matches_hand_computed_mean():
    ex = _toy_example()
    cfg = dataclasses.replace(_TINY, acoustic_weights=DECAY100)
    model = dataclasses.replace(AudioHierModel.init(_TINY, key=jax.random.PRNGKey(0)), config=cfg)

    parts = model.per_type_losses(ex)
    ce_bb = np.asarray(parts["ce_backbone"], dtype=np.float64)
    w_valid = np.asarray(parts["w_backbone"])
    tgt = np.asarray(parts["tgt_primary"])
    ce_dep = np.asarray(parts["ce_depth"], dtype=np.float64)
    w_dep = np.asarray(parts["w_depth"])

    is_sem = tgt >= SEMANTIC_LO
    alpha_bb = np.where(is_sem, cfg.alpha_semantic, cfg.alpha_text) * w_valid
    aw = np.asarray(DECAY100)
    alpha_dep = w_dep[..., None] * aw
    expected = ((ce_bb * alpha_bb).sum() + (ce_dep * alpha_dep).sum()) / (alpha_bb.sum() + alpha_dep.sum())

    assert np.isclose(float(model.compute_joint_loss(ex)), expected, rtol=1e-5)


def test_acoustic_weights_must_have_seven_entries():
    with pytest.raises(ValueError, match="acoustic_weights"):
        AudioHierConfig(acoustic_weights=(1.0, 2.0))
