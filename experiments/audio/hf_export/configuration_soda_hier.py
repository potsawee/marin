# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF configuration for the SODA hierarchical (backbone + depth) audio LM.

This file is copied verbatim into every exported checkpoint directory and
loaded via trust_remote_code, so it may import only torch/transformers —
never marin/levanter code.
"""

from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

# Llama-3-style long-rope parameters shared by backbone and depth (the values
# the runs were trained with). Written as legacy keys so transformers 4.x and
# 5.x both read them.
_DEFAULT_ROPE_THETA = 500000.0
_DEFAULT_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


class SodaHierConfig(PretrainedConfig):
    """Backbone-over-steps + depth-over-codebooks factorization of Mimi audio.

    One backbone position ("step") is a text/special token or one whole audio
    frame (8 Mimi codebooks summed at the input). The unified head predicts
    the next step's text/special/semantic id over ids 0..unified_vocab_size-1
    (identical to the flat id space); a small depth transformer predicts the
    7 acoustic codebooks within each frame.
    """

    model_type = "soda_hier"

    def __init__(
        self,
        # id space
        vocab_size: int = 144644,
        unified_vocab_size: int = 130308,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        audio_id_lo: int = 128260,
        # backbone
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 6,
        max_position_embeddings: int = 1024,
        # depth transformer
        depth_hidden_size: int = 384,
        depth_intermediate_size: int = 1536,
        depth_num_layers: int = 4,
        depth_num_heads: int = 3,
        depth_num_kv_heads: int = 3,
        depth_head_dim: int = 128,
        # shared
        rope_theta: float = _DEFAULT_ROPE_THETA,
        rope_scaling: dict | None = None,
        rms_norm_eps: float = 1e-5,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.unified_vocab_size = unified_vocab_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_id_lo = audio_id_lo
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.depth_hidden_size = depth_hidden_size
        self.depth_intermediate_size = depth_intermediate_size
        self.depth_num_layers = depth_num_layers
        self.depth_num_heads = depth_num_heads
        self.depth_num_kv_heads = depth_num_kv_heads
        self.depth_head_dim = depth_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = dict(rope_scaling) if rope_scaling else dict(_DEFAULT_ROPE_SCALING)
        self.rms_norm_eps = rms_norm_eps
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    def backbone_config(self) -> Qwen3Config:
        return Qwen3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rope_scaling=dict(self.rope_scaling),
            rms_norm_eps=self.rms_norm_eps,
            attention_bias=False,
            tie_word_embeddings=False,
            use_sliding_window=False,
            use_cache=True,
        )

    def depth_config(self) -> Qwen3Config:
        return Qwen3Config(
            vocab_size=self.num_codebooks * self.codebook_size,
            hidden_size=self.depth_hidden_size,
            intermediate_size=self.depth_intermediate_size,
            num_hidden_layers=self.depth_num_layers,
            num_attention_heads=self.depth_num_heads,
            num_key_value_heads=self.depth_num_kv_heads,
            head_dim=self.depth_head_dim,
            max_position_embeddings=self.num_codebooks,
            rope_theta=self.rope_theta,
            rope_scaling=dict(self.rope_scaling),
            rms_norm_eps=self.rms_norm_eps,
            attention_bias=False,
            tie_word_embeddings=False,
            use_sliding_window=False,
            use_cache=False,
        )
