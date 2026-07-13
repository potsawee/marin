# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P5 (per-codebook decay-weighting pilot) + the SODA-Hier HERO run.

Design log: HERO-DECISIONS.md. Runs:

    p5-hier-decay100   3e18/d768 pilot of the VoiceCraft-style geometric decay
                       weights on the audio2 corpus - directly comparable to
                       p1-hier (moshi) and p1b-hier (uniform).
    soda-hier-1b       the HERO run: ~1.11B params (d1536 backbone + dd1152/L4
                       depth) on the audio3 corpus. Requires the audio3 chunk
                       manifests; HERO_STEPS = exactly 1 epoch, pinned from the
                       aggregate manifest before launch.

Launch a run exactly as the campaign does:

    bash $SODA_ROOT/marin/experiments/audio/launchers/run_train.sh p5-hier-decay100 exp_hero.py
"""

import argparse
import dataclasses
import glob
import json
import logging
import math
import os
import shutil

from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.json_file import JsonFileTrackerConfig

from experiments.audio.exp_isoflop_headline import _common, finalize_run
from experiments.audio.isoflop_audio_target import solve_hier
from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)

# Geometric decay from the semantic weight (100, on the backbone via
# alpha_semantic) down to 1.0 at the last acoustic codebook: w_k = 100^(1-k/7).
# Precedent: VoiceCraft (arXiv:2403.16973) alpha=(5,1,0.5,0.1); see
# HERO-DECISIONS.md "Loss weighting".
DECAY100 = tuple(100.0 ** (1 - k / 7) for k in range(1, 8))

PILOT_BUDGET = 3e18  # matches P1/P1b anchors
HERO_D = 1536
# ~122M depth. CSM-anchor was dd1024 (~100M) but 1024 % 6 != 0 breaks FSDP
# sharding of the depth embed axis at 6 GPUs; dd1152 shards on 2/4/6/8
# (user-chosen over dd960 on 2026-07-12; see HERO-DECISIONS.md).
HERO_DEPTH_HIDDEN = 1152
HERO_DEPTH_LAYERS = 4
# batch=240 divides 4/6/8 GPUs (solver pow2 does not divide 6) - see
# HERO-DECISIONS.md [DEVIATION]. lr/beta2 follow the ported rules at B=240.
HERO_BATCH = 240
# Per-device microbatch: measured on 48G Ada (benches 16137635/786/821), the
# 1.09B config OOMs at microbatch 40 and 20 (unified-head CE temporaries ~15G
# at mb20 on top of sharded optimizer state); 10 fits. Divides B=240 at
# 4/6/8 GPUs (accum 6/4/3). Infra-only knob: excluded from the experiment hash.
HERO_PER_DEVICE = 10
# Exactly 1 epoch of the audio3 corpus at B=240, pinned from the verified
# aggregate manifest (2026-07-12): 148.03B flat tokens = 23.356B backbone
# steps = 396k audio hours; mix-assertion passed. Implied budget 2.12e20;
# measured wall-clock ~11.2 days on 6x Ada (19.1 on 6x Ampere).
HERO_STEPS = 95_033
HERO_BUDGET = 2.12e20


def p5_hier_decay100() -> AudioTrainConfig:
    spec, cfg = solve_hier(PILOT_BUDGET, 768)
    cfg = dataclasses.replace(cfg, acoustic_weights=DECAY100)
    config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
    return finalize_run(config, "p5-hier-decay100", spec)


def _hero_sources() -> dict[str, float]:
    """Token-proportional mixture weights over the audio3 chunk sub-sources.

    The chunk-parallel corpus build writes arm_h/{source}_{i}/ caches, each with
    a chunk manifest recording its train-token count. Weights preserve the
    campaign mix because chunks partition each source's files.
    """
    root = f"{os.environ['MARIN_PREFIX']}/audio3"
    manifests = sorted(glob.glob(f"{root}/arm_h/*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"no chunk manifests under {root}/arm_h/ - build the audio3 corpus first")
    tokens = {}
    for path in manifests:
        with open(path) as f:
            m = json.load(f)
        tokens[os.path.basename(os.path.dirname(path))] = m["train_tokens"]
    total = sum(tokens.values())
    return {name: t / total for name, t in tokens.items()}


def soda_hier_1b() -> AudioTrainConfig:
    spec, cfg = solve_hier(HERO_BUDGET, HERO_D, depth_hidden=HERO_DEPTH_HIDDEN, depth_layers=HERO_DEPTH_LAYERS)
    cfg = dataclasses.replace(cfg, acoustic_weights=DECAY100)
    spec = dataclasses.replace(
        spec,
        batch_size=HERO_BATCH,
        num_steps=HERO_STEPS,
        learning_rate=0.33 * math.sqrt(HERO_BATCH) / HERO_D,
        beta2=0.98 ** (HERO_BATCH / 128),
    )
    config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
    config = dataclasses.replace(
        config,
        data_root=f"{os.environ['MARIN_PREFIX']}/audio3",
        sources=_hero_sources(),
        trainer=dataclasses.replace(config.trainer, per_device_parallelism=HERO_PER_DEVICE),
    )
    return finalize_run(config, "soda-hier-1b", spec)


def _bench(batch: int):
    """Short REAL training run of the HERO config for step-time measurement.

    Uses the existing audio2 corpus (window shape, not content, drives step
    time), a fresh throwaway store, and 30 optimizer steps. b40 on 1 GPU
    probes the per-device operating point of the 6-GPU B=240 run; b240 on 6
    GPUs measures the real thing including all-reduce.
    """

    def build() -> AudioTrainConfig:
        spec, cfg = solve_hier(HERO_BUDGET, HERO_D, depth_hidden=HERO_DEPTH_HIDDEN, depth_layers=HERO_DEPTH_LAYERS)
        cfg = dataclasses.replace(cfg, acoustic_weights=DECAY100)
        spec = dataclasses.replace(spec, batch_size=batch, num_steps=30)
        run_id = f"bench-hero-b{batch}"
        store = f"{os.environ['SODA_ROOT']}/marin-store/smoke/{run_id}"
        config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
        trainer = dataclasses.replace(
            config.trainer,
            id=run_id,
            per_device_parallelism=HERO_PER_DEVICE,
            tracker=JsonFileTrackerConfig(output_path=store),
            checkpointer=CheckpointerConfig(base_path=f"{store}/ckpt", save_interval=None, keep=None),
        )
        return dataclasses.replace(config, trainer=trainer)

    return build


RUNS = {
    "p5-hier-decay100": p5_hier_decay100,
    "soda-hier-1b": soda_hier_1b,
    "bench-hero-b40": _bench(40),
    "bench-hero-b240": _bench(240),
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    args = parser.parse_args()
    config = RUNS[args.run]()
    if args.run.startswith("bench-"):
        # a stale checkpoint would no-op the bench; wipe only when actually running
        shutil.rmtree(f"{os.environ['SODA_ROOT']}/marin-store/smoke/{config.trainer.id}", ignore_errors=True)
    logger.info(
        "run %s: arm=%s B=%d steps=%d lr=%.5f",
        args.run,
        config.arm,
        config.trainer.train_batch_size,
        config.trainer.num_train_steps,
        config.optimizer.learning_rate,
    )
    main(config)
