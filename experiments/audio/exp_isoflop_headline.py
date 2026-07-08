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
from marin.execution.fingerprint import canonical_json, fingerprint_hash

from experiments.audio.isoflop_audio_target import RunSpec, solve_flat, solve_hier
from experiments.audio.preprocess_audio import MIX_WEIGHTS
from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)

BUDGET = 3e18
D = 768


def _base_trainer(spec: RunSpec) -> TrainerConfig:
    # Name-bearing fields (id, tracker, checkpointer) are stamped in _finalize
    # once the config hash is known.
    return TrainerConfig(
        seed=0,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=spec.batch_size,
        num_train_steps=spec.num_steps,
        steps_per_eval=10_000_000,  # NLL eval is post-hoc over checkpoints
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


def _common(spec: RunSpec) -> dict:
    return {
        "data_root": f"{os.environ['MARIN_PREFIX']}/audio2",
        "sources": dict(MIX_WEIGHTS),
        "trainer": _base_trainer(spec),
        "optimizer": _optimizer(spec),
        "flops_per_example": spec.flops_per_example,
    }


def experiment_signature(config: AudioTrainConfig) -> str:
    """Marin-canonical short hash of the EXPERIMENT-defining config.

    Includes model, data, optimizer, loss recipe (alphas), and the global
    hyperparameters (batch size, steps, seed). Deliberately EXCLUDES infra:
    device count, mesh, per-device parallelism, checkpoint paths, tracker — so
    the same experiment keeps the same hash when you swap 1<->N GPUs (the global
    batch is identical across device counts in Levanter) and resumes cleanly.
    """
    payload = {
        "arm": config.arm,
        "data_root": config.data_root,
        "sources": config.sources,
        "seq_len": config.seq_len,
        "z_loss_weight": config.z_loss_weight,
        "flat_alpha_weighted": config.flat_alpha_weighted,
        "alphas": [config.alpha_text, config.alpha_semantic, config.alpha_acoustic],
        "flat_model": config.flat_model,
        "hier_model": config.hier_model,
        "optimizer": config.optimizer,
        "batch_size": config.trainer.train_batch_size,
        "num_train_steps": config.trainer.num_train_steps,
        "seed": config.trainer.seed,
    }
    # Marin-canonical: md5 of the canonical JSON, 8 hex chars (current marin
    # convention, matching ExecutorStep.hash_id and fingerprint_hash).
    return fingerprint_hash(canonical_json(payload))


def finalize_run(config: AudioTrainConfig, base_name: str, spec: RunSpec) -> AudioTrainConfig:
    """Stamp the `{base_name}-{hash}` run name onto the tracker, checkpointer, and run id."""
    name = f"{base_name}-{experiment_signature(config)}"
    prefix = os.environ["MARIN_PREFIX"]
    trainer = dataclasses.replace(
        config.trainer,
        id=name,
        tracker=WandbConfig(
            project="soda-extension",
            name=name,
            id=name,
            tags=["isoflop", spec.arm, f"budget={spec.budget:.0e}", f"d={spec.d}"],
        ),
        checkpointer=CheckpointerConfig(base_path=f"{prefix}/audio2-runs/{name}/checkpoints"),
    )
    return dataclasses.replace(config, trainer=trainer)


def p1_flat() -> AudioTrainConfig:
    spec, cfg = solve_flat(BUDGET, D)
    config = AudioTrainConfig(arm="flat", flat_model=cfg, seq_len=4096, **_common(spec))
    return finalize_run(config, "p1-flat-uniform", spec)


def p1_hier() -> AudioTrainConfig:
    spec, cfg = solve_hier(BUDGET, D)  # AudioHierConfig defaults = Moshi alphas
    config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
    return finalize_run(config, "p1-hier-moshi", spec)


def p1b_hier_uniform() -> AudioTrainConfig:
    spec, cfg = solve_hier(BUDGET, D)
    cfg = dataclasses.replace(cfg, alpha_text=1.0, alpha_semantic=1.0, alpha_acoustic=1.0)
    config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
    return finalize_run(config, "p1b-hier-uniform", spec)


def p1c_flat_weighted() -> AudioTrainConfig:
    spec, cfg = solve_flat(BUDGET, D)
    config = AudioTrainConfig(arm="flat", flat_model=cfg, seq_len=4096, flat_alpha_weighted=True, **_common(spec))
    return finalize_run(config, "p1c-flat-moshi", spec)


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
