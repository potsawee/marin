# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shape-level construct checks for the hier campaign configs.

jax.eval_shape builds the model and traces the joint loss abstractly (no
parameters allocated, no GPU), so config-geometry bugs surface in tests
instead of at job startup. Regression: dd448 once resolved to 3 heads ->
odd head_dim 149, which crashed rotary when p2-hier-d896 launched.
"""

import haliax as hax
import jax
import jax.numpy as jnp
import pytest

from experiments.audio.data import AudioStepExample
from experiments.audio.isoflop_audio_target import HIER_STEPS, attn_heads, hier_dims
from experiments.audio.model_hier import AudioHierModel

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
