# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P1/P1b/P1c: the headline 3e18 matched pair at d=768, plus the weighting cells.

Runs (each ~7.7 GPU-hours on one RTX 6000 Ada at the planning MFU):

    p1-flat     Arm F, uniform loss (the published SODA recipe)
    p1-hier     Arm H, Moshi-weighted loss (alpha 100/100/1)   <- headline pair with p1-flat
    p1b-hier    Arm H, uniform loss (decomposes architecture vs weighting)
    p1c-flat    Arm F, Moshi-weighted loss (completes the 2x2; optional)

Launch (one GPU, submitted by the user, e.g.):

    SLURM_CPU_BIND=none CONDA_PREFIX=unused nlprun -q jag -p standard -g 1 -r 40G -c 8 \
      -n p1-hier -t 1-0 -m jagupard37 \
      'bash /nlp/scr/potsawee/workspace/soda-extension/run_train.sh p1-hier' \
      -o /nlp/scr/potsawee/workspace/soda-extension/data/runs/p1-hier.log
"""

import argparse
import dataclasses
import logging
import os

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.optim.cautious import CautiousConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.audio.isoflop_audio_target import RunSpec, solve_flat, solve_hier
from experiments.audio.preprocess_audio import MIX_WEIGHTS
from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)

BUDGET = 3e18
D = 768


def _base_trainer(spec: RunSpec, run_name: str) -> TrainerConfig:
    prefix = os.environ["MARIN_PREFIX"]
    return TrainerConfig(
        id=run_name,
        seed=0,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        tracker=WandbConfig(
            project="soda-extension",
            name=run_name,
            id=run_name,
            tags=["isoflop", spec.arm, f"budget={spec.budget:.0e}", f"d={spec.d}"],
        ),
        train_batch_size=spec.batch_size,
        num_train_steps=spec.num_steps,
        steps_per_eval=10_000_000,  # NLL eval is post-hoc over checkpoints
        checkpointer=CheckpointerConfig(base_path=f"{prefix}/audio2-runs/{run_name}/checkpoints"),
    )


def _optimizer(spec: RunSpec) -> CautiousConfig:
    return CautiousConfig(
        learning_rate=spec.learning_rate,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=0.1,
        beta1=0.95,
        beta2=spec.beta2,
        epsilon=1e-15,
        max_grad_norm=1,
        adamc_weight_decay=True,
        lr_schedule="linear",
        decay=0.2,
    )


def _common(spec: RunSpec, run_name: str) -> dict:
    return {
        "data_root": f"{os.environ['MARIN_PREFIX']}/audio2",
        "sources": dict(MIX_WEIGHTS),
        "trainer": _base_trainer(spec, run_name),
        "optimizer": _optimizer(spec),
        "flops_per_example": spec.flops_per_example,
    }


def p1_flat() -> AudioTrainConfig:
    spec, cfg = solve_flat(BUDGET, D)
    return AudioTrainConfig(arm="flat", flat_model=cfg, seq_len=4096, **_common(spec, "p1-flat-uniform"))


def p1_hier() -> AudioTrainConfig:
    spec, cfg = solve_hier(BUDGET, D)  # AudioHierConfig defaults = Moshi alphas
    return AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec, "p1-hier-moshi"))


def p1b_hier_uniform() -> AudioTrainConfig:
    spec, cfg = solve_hier(BUDGET, D)
    cfg = dataclasses.replace(cfg, alpha_text=1.0, alpha_semantic=1.0, alpha_acoustic=1.0)
    return AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec, "p1b-hier-uniform"))


def p1c_flat_weighted() -> AudioTrainConfig:
    spec, cfg = solve_flat(BUDGET, D)
    return AudioTrainConfig(
        arm="flat", flat_model=cfg, seq_len=4096, flat_alpha_weighted=True, **_common(spec, "p1c-flat-moshi")
    )


RUNS = {
    "p1-flat": p1_flat,
    "p1-hier": p1_hier,
    "p1b-hier": p1b_hier_uniform,
    "p1c-flat": p1c_flat_weighted,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    args = parser.parse_args()
    config = RUNS[args.run]()
    logger.info(
        "run %s: arm=%s B=%d steps=%d lr=%.5f",
        args.run,
        config.arm,
        config.trainer.train_batch_size,
        config.trainer.num_train_steps,
        config.optimizer.learning_rate,
    )
    main(config)
