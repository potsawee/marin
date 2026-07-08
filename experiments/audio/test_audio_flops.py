# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests pinning the accountant/solver to the old SODA sweep's grid.

The old isoflop_audio_sweep.py (audio-release-pr branch), run at its INTERNAL
forward-only budget 3e18 (= paper 9e18 after the x3 relabel), produced for
d=768: L=8, heads=6, inter=3072, B=32, steps=48244, lr~=0.00243, N~=297.7M.
Our solver at paper-9e18 must reproduce that run.
"""

import pytest

from experiments.audio.audio_flops import (
    arm_f_param_count,
    arm_h_flops_split,
    arm_h_param_count,
    gpu_hours,
)
from experiments.audio.isoflop_audio_target import flat_dims, hier_dims, solve_flat, solve_hier


def test_solver_reproduces_old_sweep_grid_point():
    spec, cfg = solve_flat(9e18, 768)
    assert (cfg.num_layers, cfg.num_heads, cfg.intermediate_dim) == (8, 6, 3072)
    assert spec.batch_size == 32
    assert spec.num_steps == pytest.approx(48244, rel=0.005)
    assert spec.learning_rate == pytest.approx(0.33 * 32**0.5 / 768, rel=1e-6)
    assert spec.beta2 == pytest.approx(0.98 ** (32 / 128), rel=1e-9)
    assert spec.params == pytest.approx(297.7e6, rel=0.01)


def test_paper_3e18_headline_point_token_budget():
    # paper-3e18 (old internal 1e18): the d=768 flattened run trains ~2.1B tokens,
    # NOT the naive-6ND 1.67B -- this pin guards the compute-axis reconciliation.
    spec, _ = solve_flat(3e18, 768)
    assert spec.tokens == pytest.approx(2.11e9, rel=0.02)
    assert spec.batch_size == 8


def test_hier_depth_rule_shapes_and_flops_share():
    spec, cfg = solve_hier(3e18, 768)
    assert (cfg.depth_hidden_dim, cfg.depth_layers) == (384, 4)
    split = arm_h_flops_split(cfg)
    assert 0.15 < split.depth_share < 0.45  # depth is a real but minority cost
    # depth params are a small fraction, in the CSM/Moshi ballpark
    hier_params = arm_h_param_count(cfg)
    flat_params = arm_f_param_count(flat_dims(768))
    assert 0.9 < hier_params / flat_params < 1.25


def test_matched_budget_means_fewer_hier_steps_tokens():
    # At the same total budget, Arm H consumes fewer step-tokens than Arm F
    # consumes flat tokens (a frame costs one backbone step + depth, not 8 full passes).
    f, _ = solve_flat(3e18, 768)
    h, _ = solve_hier(3e18, 768)
    assert h.tokens < f.tokens


def test_gpu_hours_planning_estimate():
    # 3e18 total / (0.30 MFU x 362.05 TFLOP/s bf16 dense peak) ~= 7.7 hours.
    # (The plan file's 30.9h table used the fp32 peak by mistake - 4x pessimistic.)
    assert gpu_hours(3e18) == pytest.approx(7.67, rel=0.02)
