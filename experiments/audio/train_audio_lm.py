# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared thin train main for BOTH audio-LM arms.

A fork of ``levanter.main.train_lm`` reduced to what the flattened-vs-
hierarchical comparison needs: one ``Trainer``, one optimizer build, one
seed->key split, one mixture/shuffle code path -- so the trainer is never the
variable between arms. Per-arm plug-ins are the model init, the dataset
chain, and the loss closure. No HF export, no eval harness, no adapters.

Arm F (``arm="flat"``): stock Qwen3 over 4096-token windows of the flattened
cache, EOS-derived document attention blocking, optional Moshi-style
per-target-type loss weighting (the 2x2 ablation cell).

Arm H (``arm="hier"``): AudioHierModel over 1024-step windows of the packed
step cache; loss and masks live in the model.
"""

import logging
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text.datasets import CausalLmDataset, NamedLmDataset, TokenSeqDataset
from levanter.data.text.examples import GrugLmExample
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.qwen import Qwen3Config
from levanter.optim import OptimizerConfig
from levanter.store.cache import TreeCache
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

import levanter
from levanter import callbacks

from experiments.audio.audio_vocab import EOS_ID, FULL_VOCAB, SEMANTIC_HI, SEMANTIC_LO
from experiments.audio.data import AudioStepExample, build_step_mixture
from experiments.audio.model_hier import AudioHierConfig, AudioHierModel

logger = logging.getLogger(__name__)

MIXTURE_BLOCK_SIZE = 2048


@dataclass
class AudioTrainConfig:
    arm: str = "flat"  # "flat" | "hier"
    data_root: str = ""  # e.g. $MARIN_PREFIX/audio2 (holds arm_f/ and arm_h/)
    sources: dict[str, float] = field(default_factory=dict)  # source name -> mixture weight
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # Arm F
    flat_model: Qwen3Config | None = None
    seq_len: int = 4096
    z_loss_weight: float = 1e-4
    flat_alpha_weighted: bool = False  # Moshi-style weighting on the flattened arm (2x2 cell)
    alpha_text: float = 100.0
    alpha_semantic: float = 100.0
    alpha_acoustic: float = 1.0
    # Arm H
    hier_model: AudioHierConfig | None = None
    # forward FLOPs per example from audio_flops.py (exp scripts fill this in)
    flops_per_example: float | None = None


class AlphaWeightedDataset(MappedAsyncDataset[GrugLmExample, GrugLmExample]):
    """Rescales per-position loss weights by the Moshi alpha of each position's TARGET id."""

    def __init__(self, dataset, alpha_text: float, alpha_semantic: float, alpha_acoustic: float):
        @eqx.filter_jit
        def _reweight(ex: GrugLmExample) -> GrugLmExample:
            tgt = jnp.roll(ex.tokens, -1)
            is_sem = (tgt >= SEMANTIC_LO) & (tgt < SEMANTIC_HI)
            is_aco = tgt >= SEMANTIC_HI
            alpha = jnp.where(is_sem, alpha_semantic, jnp.where(is_aco, alpha_acoustic, alpha_text))
            return eqx.tree_at(lambda e: e.loss_weight, ex, ex.loss_weight * alpha)

        super().__init__(dataset, _reweight)


def _flat_train_dataset(config: AudioTrainConfig, Pos: Axis, data_key):
    shuffle_key, mix_key = jrandom.split(data_key)
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32)}
    parts = {}
    for i, (name, _) in enumerate(sorted(config.sources.items())):
        cache = TreeCache.load(f"{config.data_root}/arm_f/{name}/train", exemplar)
        parts[name] = TokenSeqDataset(cache, Pos.size).shuffle(jrandom.fold_in(shuffle_key, i))
    mixed = MixtureDataset(parts, config.sources, MIXTURE_BLOCK_SIZE, key=mix_key)
    causal = CausalLmDataset(mixed, Pos, eos_id=EOS_ID, block_cross_document_attention=True)
    if config.flat_alpha_weighted:
        causal = AlphaWeightedDataset(causal, config.alpha_text, config.alpha_semantic, config.alpha_acoustic)
    return NamedLmDataset(causal, Pos)


def _hier_train_dataset(config: AudioTrainConfig, data_key):
    cache_dirs = {name: f"{config.data_root}/arm_h/{name}/train" for name in config.sources}
    return build_step_mixture(cache_dirs, config.sources, key=data_key, block_size=MIXTURE_BLOCK_SIZE)


def main(config: AudioTrainConfig):
    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    if config.arm == "flat":
        assert config.flat_model is not None

        def loss_function(model: LmHeadModel, example: LmExample, *, key=None):
            return model.compute_next_token_loss(example, key=key, logsumexp_weight=config.z_loss_weight)

    elif config.arm == "hier":
        assert config.hier_model is not None

        def loss_function(model: AudioHierModel, example: AudioStepExample, *, key=None):
            return model.compute_joint_loss(example, key=key)

    else:
        raise ValueError(f"unknown arm {config.arm!r}")

    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.arm == "flat":
            Pos = config.flat_model.max_Pos.resize(config.seq_len)
            Vocab = round_axis_for_partitioning(Axis("vocab", FULL_VOCAB), trainer.parameter_axis_mapping)
            train_dataset = _flat_train_dataset(config, Pos, data_key)
            model_init = lambda: config.flat_model.build(Vocab, key=model_key)  # noqa: E731
            pos_size = Pos.size
        else:
            train_dataset = _hier_train_dataset(config, data_key)
            model_init = lambda: AudioHierModel.init(config.hier_model, key=model_key)  # noqa: E731
            pos_size = config.hier_model.max_steps

        state = trainer.initial_state(training_key, model_init=model_init)
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        if config.flops_per_example is not None:
            trainer.add_hook(
                callbacks.log_performance_stats(pos_size, trainer.config.batch_schedule, 3 * config.flops_per_example),
                every=1,
            )

        train_loader = trainer.data_loader(train_dataset)
        train_loader = train_loader.iter_from_step(int(state.step))
        trainer.train(state, train_loader)

    trainer.tracker.finish()
