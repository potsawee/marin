# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solve isoflop target runs for both arms (port of the old SODA derivation rules).

Given a TOTAL-FLOPs budget (paper convention = 3x forward) and a backbone width
d, derive the model dims, batch size, LR, beta2, and step count exactly the way
`experiments/audio/isoflop_audio_sweep.py` did on the audio-release-pr branch:

    L      = round(d / (64 + 4*log2(d) - 7))
    heads  = max(1, d // 128); kv = heads; inter = 4d
    B      = nearest-pow2 of [fwd_budget / (fwd_per_unit * 2^16 * seq)], min 8,
             halved while lr = 0.33*sqrt(B)/d exceeds 0.01
    beta2  = 0.98 ** (B / 128)
    steps  = round(fwd_budget / (fwd_per_unit * B * seq))

For Arm H, d is the BACKBONE width; the depth transformer follows the rule
(half width, 4 layers) unless explicit dims are given (the depth ablation), and
`fwd_per_unit` is per backbone STEP with seq = L_STEPS = 1024.

Run as a script to print the campaign table:

    uv run python experiments/audio/isoflop_audio_target.py
"""

import math
from dataclasses import dataclass

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config

from experiments.audio.audio_flops import (
    TRAIN_MULT,
    arm_f_fwd_per_token,
    arm_f_param_count,
    arm_h_flops_split,
    arm_h_fwd_per_step,
    arm_h_param_count,
    gpu_hours,
)
from experiments.audio.model_hier import AudioHierConfig

FLAT_SEQ_LEN = 4096
HIER_STEPS = 1024
STEPS_PER_RUN_ANCHOR = 2**16  # the old sweep's batch-sizing anchor
LR_CONST = 0.33
LR_CAP = 0.01
MIN_BATCH = 8


@dataclass(frozen=True)
class RunSpec:
    """Everything an exp script needs to launch one isoflop run."""

    arm: str  # "flat" | "hier"
    budget: float  # TOTAL training FLOPs (paper convention)
    d: int  # backbone width
    batch_size: int
    num_steps: int
    learning_rate: float
    beta2: float
    tokens: float  # flat tokens (Arm F) or backbone steps (Arm H) consumed
    params: int
    flops_per_example: float  # forward FLOPs per training example (for MFU logging)
    name: str


def flat_dims(d: int) -> Qwen3Config:
    num_layers = round(d / (64 + 4 * math.log2(d) - 7))
    heads = max(1, d // 128)
    return Qwen3Config(
        max_seq_len=FLAT_SEQ_LEN,
        hidden_dim=d,
        intermediate_dim=4 * d,
        num_layers=num_layers,
        num_heads=heads,
        num_kv_heads=heads,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=False,
    )


def hier_dims(d: int, *, depth_hidden: int | None = None, depth_layers: int = 4) -> AudioHierConfig:
    num_layers = round(d / (64 + 4 * math.log2(d) - 7))
    heads = max(1, d // 128)
    d_d = depth_hidden if depth_hidden is not None else d // 2
    return AudioHierConfig(
        max_steps=HIER_STEPS,
        hidden_dim=d,
        intermediate_dim=4 * d,
        num_layers=num_layers,
        num_heads=heads,
        num_kv_heads=heads,
        depth_hidden_dim=d_d,
        depth_intermediate_dim=4 * d_d,
        depth_layers=depth_layers,
        depth_heads=max(1, d_d // 128),
        depth_kv_heads=max(1, d_d // 128),
        rope=Llama3RotaryEmbeddingsConfig(),
    )


def _solve_batch_and_steps(fwd_budget: float, fwd_per_unit: float, seq: int, d: int) -> tuple[int, int, float]:
    batch_exact = fwd_budget / (fwd_per_unit * STEPS_PER_RUN_ANCHOR * seq)
    batch = max(MIN_BATCH, 2 ** round(math.log2(max(batch_exact, 1e-9))))
    lr = LR_CONST * math.sqrt(batch) / d
    while lr > LR_CAP:
        batch //= 2
        lr = LR_CONST * math.sqrt(batch) / d
    steps = round(fwd_budget / (fwd_per_unit * batch * seq))
    return batch, steps, lr


def solve_flat(budget: float, d: int) -> tuple[RunSpec, Qwen3Config]:
    cfg = flat_dims(d)
    fwd = arm_f_fwd_per_token(cfg)
    batch, steps, lr = _solve_batch_and_steps(budget / TRAIN_MULT, fwd, FLAT_SEQ_LEN, d)
    params = arm_f_param_count(cfg)
    spec = RunSpec(
        arm="flat",
        budget=budget,
        d=d,
        batch_size=batch,
        num_steps=steps,
        learning_rate=lr,
        beta2=0.98 ** (batch / 128),
        tokens=float(steps) * batch * FLAT_SEQ_LEN,
        params=params,
        flops_per_example=fwd * FLAT_SEQ_LEN,
        name=f"flat-isoflop-{budget:.0e}-{params / 1e6:.0f}M-d{d}",
    )
    return spec, cfg


def solve_hier(budget: float, d: int, *, depth_hidden: int | None = None, depth_layers: int = 4) -> tuple[RunSpec, AudioHierConfig]:
    cfg = hier_dims(d, depth_hidden=depth_hidden, depth_layers=depth_layers)
    fwd = arm_h_fwd_per_step(cfg)
    batch, steps, lr = _solve_batch_and_steps(budget / TRAIN_MULT, fwd, HIER_STEPS, d)
    params = arm_h_param_count(cfg)
    spec = RunSpec(
        arm="hier",
        budget=budget,
        d=d,
        batch_size=batch,
        num_steps=steps,
        learning_rate=lr,
        beta2=0.98 ** (batch / 128),
        tokens=float(steps) * batch * HIER_STEPS,
        params=params,
        flops_per_example=fwd * HIER_STEPS,
        name=f"hier-isoflop-{budget:.0e}-{params / 1e6:.0f}M-d{d}-dd{cfg.depth_hidden_dim}L{depth_layers}",
    )
    return spec, cfg


def campaign_table() -> list[RunSpec]:
    """The planned campaign (P1/P2/P4 grid + P3 depth ablation), for inspection."""
    specs = []
    for budget in (1e18, 3e18):
        for d in (512, 768, 896):
            if budget == 1e18 and d != 768:
                continue  # P4 is the d=768 anchor pair only
            specs.append(solve_flat(budget, d)[0])
            specs.append(solve_hier(budget, d)[0])
    # P3: depth ablation at 3e18, d=768 (small / large; medium == the sweep point)
    specs.append(solve_hier(3e18, 768, depth_hidden=256, depth_layers=2)[0])
    specs.append(solve_hier(3e18, 768, depth_hidden=512, depth_layers=6)[0])
    return specs


if __name__ == "__main__":
    print(f"{'name':52s} {'params':>8s} {'B':>4s} {'steps':>7s} {'lr':>9s} {'tokens':>9s} {'GPU-h':>6s}")
    for spec in campaign_table():
        print(
            f"{spec.name:52s} {spec.params / 1e6:7.1f}M {spec.batch_size:4d} {spec.num_steps:7d}"
            f" {spec.learning_rate:9.5f} {spec.tokens / 1e9:8.2f}B {gpu_hours(spec.budget):6.1f}"
        )
    hier_768 = hier_dims(768)
    split = arm_h_flops_split(hier_768)
    print(f"\nArm H d=768 fwd split: backbone {split.backbone / 1e6:.0f}M, depth {split.depth / 1e6:.0f}M "
          f"({split.depth_share:.0%} of per-step FLOPs)")
