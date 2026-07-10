# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve campaign run names to trained configs and on-disk checkpoint paths.

The three exp scripts' RUNS registries are the single source of truth for run
geometry; ``finalize_run`` stamps ``trainer.id`` (the hashed run-dir name) and
the checkpointer base path, so everything here is derived, never re-specified.
"""

import os
import re
from dataclasses import dataclass

from experiments.audio import exp_depth_ablation, exp_isoflop_headline, exp_isoflop_sweep
from experiments.audio.train_audio_lm import AudioTrainConfig

ALL_RUNS = {**exp_isoflop_headline.RUNS, **exp_isoflop_sweep.RUNS, **exp_depth_ablation.RUNS}

_STEP_DIR = re.compile(r"^step-(\d+)$")


@dataclass(frozen=True)
class RunHandle:
    run: str  # registry key, e.g. "p1-flat"
    arm: str  # "flat" | "hier"
    config: AudioTrainConfig
    run_id: str  # hashed dir name, e.g. "p1-flat-uniform-b78282a0"
    ckpt_root: str  # .../audio2-runs/<run_id>/checkpoints/<run_id>
    step: int
    hf_out_dir: str  # .../audio2-runs/<run_id>/hf/step-<step>

    @property
    def step_dir(self) -> str:
        return f"{self.ckpt_root}/step-{self.step}"


def latest_step(ckpt_root: str) -> int:
    steps = [int(m.group(1)) for name in os.listdir(ckpt_root) if (m := _STEP_DIR.match(name))]
    if not steps:
        raise FileNotFoundError(f"no step-N checkpoints under {ckpt_root}")
    return max(steps)


def resolve(run: str, step: int | None = None) -> RunHandle:
    if "MARIN_PREFIX" not in os.environ:
        raise RuntimeError("MARIN_PREFIX not set — source soda-extension/env.sh first")
    if run not in ALL_RUNS:
        raise KeyError(f"unknown run {run!r}; known: {sorted(ALL_RUNS)}")
    config = ALL_RUNS[run]()
    run_id = config.trainer.id
    ckpt_root = f"{config.trainer.checkpointer.expanded_path(run_id)}"
    if step is None:
        step = latest_step(ckpt_root)
    run_root = os.path.dirname(os.path.dirname(ckpt_root))  # strips /checkpoints/<run_id>
    return RunHandle(
        run=run,
        arm=config.arm,
        config=config,
        run_id=run_id,
        ckpt_root=ckpt_root,
        step=step,
        hf_out_dir=f"{run_root}/hf/step-{step}",
    )


def flat_runs() -> list[str]:
    return [r for r in ALL_RUNS if ALL_RUNS[r]().arm == "flat"]


def hier_runs() -> list[str]:
    return [r for r in ALL_RUNS if ALL_RUNS[r]().arm == "hier"]
