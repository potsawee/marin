# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FLOPs accountants and parameter counts for the two audio-LM arms.

Convention (matches the SODA paper's compute axis): a run's budget is TOTAL
training FLOPs = ``TRAIN_MULT`` x forward FLOPs. The old sweep script's
internal budgets were forward-only and the paper relabeled them x3, so
``arm_f_fwd_per_token(cfg) * tokens * TRAIN_MULT`` plots directly onto the
paper's isoflop figures.

Arm H charges the depth transformer at EVERY backbone step (text steps
included), matching the ``depth_on_all_steps=True`` v1 implementation exactly.
"""

from dataclasses import dataclass

import jax
import jax.random as jrandom
from haliax import Axis
from levanter.models.qwen import Qwen3Config
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.audio.audio_vocab import CODEBOOK_SIZE, FULL_VOCAB, NUM_CODEBOOKS, UNIFIED_VOCAB
from experiments.audio.model_hier import AudioHierConfig, AudioHierModel

TRAIN_MULT = 3  # forward + backward ~= 3x forward
# RTX 6000 Ada dense bf16 peak (see lib/fray/src/fray/device_flops.py) and the
# planning MFU assumption used for wall-clock estimates only.
RTX6000_ADA_BF16 = 362.05e12
PLANNING_MFU = 0.30


def arm_f_fwd_per_token(cfg: Qwen3Config, vocab_size: int = FULL_VOCAB, seq_len: int = 4096) -> float:
    """Forward FLOPs per flattened token (the old sweep's accountant, verbatim)."""
    return lm_flops_per_token(
        cfg.hidden_dim,
        cfg.intermediate_dim,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.num_heads,
        seq_len,
        vocab_size,
        glu=True,
    )


def arm_h_fwd_per_step(cfg: AudioHierConfig) -> float:
    """Forward FLOPs per backbone step = backbone + depth (charged every step)."""
    backbone = lm_flops_per_token(
        cfg.hidden_dim,
        cfg.intermediate_dim,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.num_heads,
        cfg.max_steps,
        UNIFIED_VOCAB,  # the unified head is the backbone's "lm_head"
        glu=True,
    )
    depth_per_token = lm_flops_per_token(
        cfg.depth_hidden_dim,
        cfg.depth_intermediate_dim,
        cfg.depth_layers,
        cfg.depth_kv_heads,
        cfg.depth_heads,
        NUM_CODEBOOKS,
        0,  # per-codebook heads accounted separately below
        glu=True,
    )
    depth = NUM_CODEBOOKS * depth_per_token
    acoustic_heads = (NUM_CODEBOOKS - 1) * 2 * cfg.depth_hidden_dim * CODEBOOK_SIZE
    bd_proj = 2 * cfg.hidden_dim * cfg.depth_hidden_dim
    return backbone + depth + acoustic_heads + bd_proj


def _param_count_abstract(init_fn) -> int:
    shapes = jax.eval_shape(init_fn)
    return sum(leaf.size for leaf in jax.tree.leaves(shapes) if isinstance(leaf, jax.ShapeDtypeStruct))


def arm_f_param_count(cfg: Qwen3Config, vocab_size: int = FULL_VOCAB) -> int:
    """Exact parameter count of the flattened model (embeddings included, untied)."""
    return _param_count_abstract(lambda: cfg.build(Axis("vocab", vocab_size), key=jrandom.PRNGKey(0)))


def arm_h_param_count(cfg: AudioHierConfig) -> int:
    """Exact parameter count of the hierarchical model."""
    return _param_count_abstract(lambda: AudioHierModel.init(cfg, key=jrandom.PRNGKey(0)))


@dataclass(frozen=True)
class FlopsSplit:
    """Where an Arm H step's forward FLOPs go (for reporting/analysis)."""

    backbone: float
    depth: float

    @property
    def depth_share(self) -> float:
        return self.depth / (self.backbone + self.depth)


def arm_h_flops_split(cfg: AudioHierConfig) -> FlopsSplit:
    total = arm_h_fwd_per_step(cfg)
    backbone = lm_flops_per_token(
        cfg.hidden_dim,
        cfg.intermediate_dim,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.num_heads,
        cfg.max_steps,
        UNIFIED_VOCAB,
        glu=True,
    )
    return FlopsSplit(backbone=backbone, depth=total - backbone)


def gpu_hours(total_flops: float, mfu: float = PLANNING_MFU) -> float:
    """Estimated wall-clock GPU-hours on one RTX 6000 Ada at the planning MFU."""
    return total_flops / (mfu * RTX6000_ADA_BF16) / 3600
