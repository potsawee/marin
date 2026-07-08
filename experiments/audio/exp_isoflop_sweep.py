# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P2/P4: 3e18 backbone sweep (d=512, 896; both arms) and the 1e18 anchor pair at d=768.

Runs (P1's d=768 pair lives in exp_isoflop_headline.py):

    p2-flat-d512   p2-hier-d512   p2-flat-d896   p2-hier-d896     (3e18, ~7.7 GPU-h each)
    p4-flat-d768   p4-hier-d768                                    (1e18, ~2.6 GPU-h each)

Launch like the headline runs:

    SLURM_CPU_BIND=none CONDA_PREFIX=unused nlprun -q jag -p standard -g 1 -r 40G -c 8 \
      -n p2-hier-d512 -t 1-0 -m jagupard38 \
      'bash /nlp/scr/potsawee/workspace/soda-extension/run_train.sh p2-hier-d512 exp_isoflop_sweep.py' \
      -o /nlp/scr/potsawee/workspace/soda-extension/data/runs/p2-hier-d512.log
"""

import argparse
import logging

from experiments.audio.exp_isoflop_headline import _common, finalize_run
from experiments.audio.isoflop_audio_target import solve_flat, solve_hier
from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)


def _flat(budget: float, d: int, tag: str) -> AudioTrainConfig:
    spec, cfg = solve_flat(budget, d)
    config = AudioTrainConfig(arm="flat", flat_model=cfg, seq_len=4096, **_common(spec))
    return finalize_run(config, f"{tag}-flat-d{d}", spec)


def _hier(budget: float, d: int, tag: str) -> AudioTrainConfig:
    spec, cfg = solve_hier(budget, d)
    config = AudioTrainConfig(arm="hier", hier_model=cfg, **_common(spec))
    return finalize_run(config, f"{tag}-hier-d{d}", spec)


RUNS = {
    "p2-flat-d512": lambda: _flat(3e18, 512, "p2"),
    "p2-hier-d512": lambda: _hier(3e18, 512, "p2"),
    "p2-flat-d896": lambda: _flat(3e18, 896, "p2"),
    "p2-hier-d896": lambda: _hier(3e18, 896, "p2"),
    "p4-flat-d768": lambda: _flat(1e18, 768, "p4"),
    "p4-hier-d768": lambda: _hier(1e18, 768, "p4"),
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    args = parser.parse_args()
    main(RUNS[args.run]())
