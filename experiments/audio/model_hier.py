# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CSM/Moshi-style hierarchical audio LM (Arm H).

A **backbone** decoder (the same Qwen3 stack the flattened arm uses) runs over
"steps": one text/special token or one whole audio frame per position, the
frame input being the sum of its 8 codebook embeddings. A unified head over
text + specials + semantic-codebook predicts each step's primary token. A small
**depth** decoder (same Qwen3 blocks, sequence axis = the 8 codebook slots,
batched over steps) predicts the next frame's 7 acoustic codebooks
autoregressively, conditioned on the backbone hidden state and teacher-forced
codebook prefix, with per-codebook output heads (CSM factorization: semantic is
the backbone's job).

The joint loss counts one CE term per underlying token -- identical term count
to the flattened arm over the same document -- with Moshi-style alpha--weighting
(``sum  alpha-*ce / sum  alpha-*w``) as a config knob. ``per_type_losses`` exposes unreduced
per-term CE for the comparable-NLL evaluator.
"""

from dataclasses import dataclass, field

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from levanter.layers.attention import AttentionMask
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaTransformer
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.models.qwen import Qwen3Config

from experiments.audio.audio_vocab import (
    AUDIO_ID_LO,
    CODEBOOK_SIZE,
    FULL_VOCAB,
    NUM_AUDIO_TOKENS,
    NUM_CODEBOOKS,
    SEMANTIC_HI,
    SEMANTIC_LO,
    UNIFIED_VOCAB,
)
from experiments.audio.data import PAD, AudioStepExample


@dataclass(frozen=True)
class AudioHierConfig:
    """Backbone + depth dimensions and the training-loss recipe."""

    max_steps: int = 1024
    # backbone (sized like the flattened Qwen3 grid points)
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    num_layers: int = 8
    num_heads: int = 6
    num_kv_heads: int = 6
    # depth transformer (rule: half width, 4 layers)
    depth_hidden_dim: int = 384
    depth_intermediate_dim: int = 1536
    depth_layers: int = 4
    depth_heads: int = 3
    depth_kv_heads: int = 3
    rope: RotaryEmbeddingsConfig = field(default_factory=Llama3RotaryEmbeddingsConfig)
    # loss recipe (Moshi defaults: alpha-=100 on text+semantic, 1 on acoustic)
    alpha_text: float = 100.0
    alpha_semantic: float = 100.0
    alpha_acoustic: float = 1.0
    z_loss_weight: float | None = 1e-4

    def backbone_config(self) -> Qwen3Config:
        return Qwen3Config(
            max_seq_len=self.max_steps,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rope=self.rope,
            tie_word_embeddings=False,
        )

    def depth_config(self) -> Qwen3Config:
        # Stacked cannot scan an empty layer axis; a 0-layer depth stack (the
        # degenerate-depth ablation) must use the sequential BlockSeq path.
        return Qwen3Config(
            scan_layers=self.depth_layers > 0,
            max_seq_len=NUM_CODEBOOKS,
            hidden_dim=self.depth_hidden_dim,
            intermediate_dim=self.depth_intermediate_dim,
            num_layers=self.depth_layers,
            num_heads=self.depth_heads,
            num_kv_heads=self.depth_kv_heads,
            rope=self.rope,
            tie_word_embeddings=False,
        )

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def DepthEmbed(self) -> Axis:
        return Axis("depth_embed", self.depth_hidden_dim)


FullVocab = Axis("full_vocab", FULL_VOCAB)
Unified = Axis("unified", UNIFIED_VOCAB)
AudioVocab = Axis("audio_vocab", NUM_AUDIO_TOKENS)
Codebook = Axis("codebook", NUM_CODEBOOKS)
Acoustic = Axis("acoustic", NUM_CODEBOOKS - 1)
CB = Axis("cb", CODEBOOK_SIZE)


class AudioHierModel(eqx.Module):
    config: AudioHierConfig = eqx.field(static=True)
    embed: hnn.Embedding  # FullVocab x Embed -- shared backbone input table
    backbone: LlamaTransformer
    unified_head: hnn.Linear  # Embed -> Unified
    bd_proj: hnn.Linear  # Embed -> DepthEmbed (backbone hidden into the depth stack)
    depth_embed: hnn.Embedding  # AudioVocab x DepthEmbed -- depth teacher-forcing inputs
    depth: LlamaTransformer
    acoustic_heads: NamedArray  # Acoustic x DepthEmbed x CB -- per-codebook output heads

    @classmethod
    def init(cls, config: AudioHierConfig, *, key) -> "AudioHierModel":
        k_e, k_b, k_u, k_p, k_de, k_d, k_h = jrandom.split(key, 7)
        embed = hnn.Embedding.init(FullVocab, config.Embed, key=k_e)
        backbone = LlamaTransformer.init(config.backbone_config(), key=k_b)
        unified_head = hnn.Linear.init(In=config.Embed, Out=Unified, key=k_u, use_bias=False, out_first=True)
        bd_proj = hnn.Linear.init(In=config.Embed, Out=config.DepthEmbed, key=k_p, use_bias=False, out_first=True)
        depth_embed = hnn.Embedding.init(AudioVocab, config.DepthEmbed, key=k_de)
        depth = LlamaTransformer.init(config.depth_config(), key=k_d)
        acoustic_heads = hax.random.truncated_normal(k_h, (Acoustic, config.DepthEmbed, CB), lower=-3, upper=3) * (0.02)
        return cls(config, embed, backbone, unified_head, bd_proj, depth_embed, depth, acoustic_heads)

    def backbone_hidden(self, ex: AudioStepExample) -> NamedArray:
        """Embed steps (summing frame codebooks) and run the backbone. -> {.., position, embed}"""
        codes = ex.codes
        seg_ids = ex.seg_ids
        valid_slot = codes.array != PAD
        emb_ids = hax.named(jnp.where(valid_slot, codes.array, 0), codes.axes)
        per_slot = self.embed.embed(emb_ids)  # {.., position, codebook, embed}
        per_slot = per_slot * hax.named(valid_slot, codes.axes).astype(per_slot.dtype)
        x = per_slot.sum(Codebook)
        mask = AttentionMask.causal().with_segment_ids(seg_ids)
        return self.backbone(x, mask, key=None)

    def _depth_hidden(self, h: NamedArray, tgt_codes: NamedArray) -> NamedArray:
        """Depth stack over the 8 codebook slots of the TARGET frame, teacher-forced.

        Slot k's input = projected backbone hidden + (for k>=1) the embedding of the
        target frame's codebook k-1 token. Returns hidden states {.., position, codebook, depth_embed}.
        """
        cond = self.bd_proj(h)  # {.., position, depth_embed}
        # teacher-forced inputs: audio-vocab index of target codebooks 0..6 feed slots 1..7
        audio_idx = jnp.clip(tgt_codes.array - AUDIO_ID_LO, 0, NUM_AUDIO_TOKENS - 1)
        prefix = self.depth_embed.embed(hax.named(audio_idx, tgt_codes.axes))  # {.., position, codebook, depth_embed}
        shifted = hax.roll(prefix, 1, Codebook)  # slot k sees codebook k-1; slot 0 sees garbage ->
        slot0 = hax.arange(Codebook) == 0
        shifted = shifted * (1 - slot0.astype(shifted.dtype))  # zero the wrapped slot-0 input
        x = cond.broadcast_axis(Codebook) + shifted
        # depth attends over the 8 slots; steps become a batch axis
        x = x.rename({"position": "step", "codebook": "position", "depth_embed": "embed"})
        out = self.depth(x, AttentionMask.causal(), key=None)
        return out.rename({"position": "codebook", "step": "position", "embed": "depth_embed"})

    def per_type_losses(self, ex: AudioStepExample) -> dict[str, jax.Array]:
        """Unreduced per-term CE and masks/targets for the loss and the evaluator.

        Backbone terms: one per step transition (text, special, or semantic target).
        Depth terms: 7 per frame-target transition. Weights already exclude padding,
        cross-document transitions, and the final (wrapped) position.
        """
        codes = ex.codes
        seg_ids = ex.seg_ids
        Pos = codes.resolve_axis("position")
        h = self.backbone_hidden(ex)

        tgt_codes = hax.roll(codes, -1, "position")
        tgt_seg = hax.roll(seg_ids, -1, "position")
        not_last = hax.arange(Pos) < Pos.size - 1
        w_valid = (seg_ids.array != PAD) & (tgt_seg.array == seg_ids.array) & not_last.array

        tgt_primary = tgt_codes["codebook", 0]
        tgt_is_frame = (tgt_primary.array >= SEMANTIC_LO) & (tgt_primary.array < SEMANTIC_HI)

        # backbone CE over the unified head (fused: never materializes 130k logits)
        bb_targets = hax.named(jnp.where(w_valid, tgt_primary.array, 0), tgt_primary.axes)
        ce_bb = fused_cross_entropy_loss_and_logsumexp_penalty(
            h,
            self.unified_head.weight,
            Contract="embed",
            Label="unified",
            target_y=bb_targets,
            reduction=None,
            logsumexp_weight=self.config.z_loss_weight,
        )

        # depth CE per acoustic codebook (fused per head; 2048-way)
        d_out = self._depth_hidden(h, tgt_codes)
        w_depth = w_valid & tgt_is_frame
        ce_dep = []
        for k in range(1, NUM_CODEBOOKS):
            head_k = self.acoustic_heads["acoustic", k - 1]  # {depth_embed, cb}
            tgt_k = tgt_codes["codebook", k].array - AUDIO_ID_LO - k * CODEBOOK_SIZE
            tgt_k = hax.named(jnp.where(w_depth, tgt_k, 0), seg_ids.axes)
            ce_k = fused_cross_entropy_loss_and_logsumexp_penalty(
                d_out["codebook", k - 1],
                head_k,
                Contract="depth_embed",
                Label="cb",
                target_y=tgt_k,
                reduction=None,
                logsumexp_weight=None,
            )
            ce_dep.append(ce_k.array)

        return {
            "ce_backbone": ce_bb.array,
            "w_backbone": w_valid,
            "tgt_primary": tgt_primary.array,
            "ce_depth": jnp.stack(ce_dep, axis=-1),  # (.., position, 7)
            "w_depth": w_depth,
        }

    def compute_joint_loss(self, ex: AudioStepExample, *, key=None) -> jax.Array:
        """alpha--weighted mean CE over all terms (the training objective)."""
        parts = self.per_type_losses(ex)
        cfg = self.config
        tgt = parts["tgt_primary"]
        is_sem = (tgt >= SEMANTIC_LO) & (tgt < SEMANTIC_HI)
        # text and the four block-marker specials share alpha-_text
        alpha_bb = jnp.where(is_sem, cfg.alpha_semantic, cfg.alpha_text)
        w_bb = parts["w_backbone"].astype(jnp.float32) * alpha_bb
        w_dep = parts["w_depth"].astype(jnp.float32)[..., None] * cfg.alpha_acoustic

        num = (parts["ce_backbone"] * w_bb).sum() + (parts["ce_depth"] * w_dep).sum()
        # every frame contributes 7 acoustic CE terms to the weight mass
        denom = w_bb.sum() + w_dep.sum() * (NUM_CODEBOOKS - 1)
        return num / jnp.maximum(denom, 1.0)


def param_count(model: AudioHierModel) -> int:
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    return sum(leaf.size for leaf in leaves)
