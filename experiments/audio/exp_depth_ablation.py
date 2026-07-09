# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P3: depth-size ablation at 3e18, backbone d=768 (medium == p1-hier).

    p3-small   depth 256-wide, 2 layers  (~3% of params)
    p3-large   depth 512-wide, 6 layers  (~13% of params)

Launch (user submits): launchers/launch_campaign.sh submits these with the
rest of the campaign. To hand-launch one run, wrap in nlprun exactly as that
script does:

    bash $SODA_ROOT/marin/experiments/audio/launchers/run_train.sh p3-small exp_depth_ablation.py
"""

import argparse
import logging

from experiments.audio.exp_isoflop_headline import _common, finalize_run
from experiments.audio.isoflop_audio_target import solve_hier
from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)


def _ablation(depth_hidden: int, depth_layers: int, tag: str) -> AudioTrainConfig:
    spec, cfg = solve_hier(3e18, 768, depth_hidden=depth_hidden, depth_layers=depth_layers)
    config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
    return finalize_run(config, tag, spec)


RUNS = {
    "p3-small": lambda: _ablation(256, 2, "p3-hier-dd256L2"),
    "p3-large": lambda: _ablation(512, 6, "p3-hier-dd512L6"),
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    args = parser.parse_args()
    main(RUNS[args.run]())
