# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert an Arm-H (hier) levanter checkpoint to a trust_remote_code HF dir.

Writes <run_root>/hf/step-N/ with model.safetensors, config.json (legacy rope
keys + auto_map), the two modeling .py files, and the tokenizer — loadable via
AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True).

Usage (CPU, marin venv, env.sh sourced):
    python -m experiments.audio.hf_export.convert_hier_to_hf --run p1-hier [--step N]
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import jax.random as jrandom
import numpy as np
import torch
from haliax.state_dict import save_state_dict, to_torch_compatible_state_dict
from levanter.checkpoint import load_checkpoint
from levanter.utils.jax_utils import local_cpu_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.audio.audio_vocab import (
    AUDIO_ID_LO,
    BOS_ID,
    CODEBOOK_SIZE,
    EOS_ID,
    FULL_VOCAB,
    NUM_CODEBOOKS,
    TOKENIZER_ID,
    UNIFIED_VOCAB,
)
from experiments.audio.hf_export.configuration_soda_hier import SodaHierConfig
from experiments.audio.hf_export.run_registry import RunHandle, hier_runs, resolve
from experiments.audio.model_hier import AudioHierConfig, AudioHierModel

logger = logging.getLogger(__name__)


def soda_hier_config_from(cfg: AudioHierConfig) -> SodaHierConfig:
    return SodaHierConfig(
        vocab_size=FULL_VOCAB,
        unified_vocab_size=UNIFIED_VOCAB,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        audio_id_lo=AUDIO_ID_LO,
        hidden_size=cfg.hidden_dim,
        intermediate_size=cfg.intermediate_dim,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.max_steps,
        depth_hidden_size=cfg.depth_hidden_dim,
        depth_intermediate_size=cfg.depth_intermediate_dim,
        depth_num_layers=cfg.depth_layers,
        depth_num_heads=cfg.depth_heads,
        depth_num_kv_heads=cfg.depth_kv_heads,
        depth_head_dim=cfg.depth_hidden_dim // cfg.depth_heads,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
    )


def torch_state_dict_from_jax(model: AudioHierModel) -> dict[str, np.ndarray]:
    """Map the haliax state dict onto SodaHierForCausalLM's module names."""
    sd = to_torch_compatible_state_dict(model)
    out: dict[str, np.ndarray] = {}
    for key, value in sd.items():
        arr = np.asarray(value)
        if key == "embed.weight":
            out["backbone.embed_tokens.weight"] = arr
        elif key == "depth_embed.weight":
            out["depth.embed_tokens.weight"] = arr
        elif key == "acoustic_heads":
            # stacked {acoustic, depth_embed, cb}; torch Linear wants (cb, depth_embed)
            for k in range(arr.shape[0]):
                out[f"acoustic_heads.{k}.weight"] = np.ascontiguousarray(arr[k].T)
        else:
            # unified_head/bd_proj (out_first Linear) and backbone./depth. blocks
            # already carry HF-compatible names and (out, in) layouts.
            out[key] = arr
    return out


def convert_run(run: str, step: int | None = None) -> RunHandle:
    handle = resolve(run, step)
    if handle.arm != "hier":
        raise ValueError(f"{run} is arm={handle.arm}; use export_flat_hf.py for flat runs")
    logger.info("converting %s (%s step %d) -> %s", run, handle.run_id, handle.step, handle.hf_out_dir)

    with local_cpu_mesh():
        model = AudioHierModel.init(handle.config.hier_model, key=jrandom.PRNGKey(0))
        model = load_checkpoint(model, handle.step_dir, subpath="model")
        state_dict = torch_state_dict_from_jax(model)

    out = Path(handle.hf_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_state_dict(state_dict, str(out / "model.safetensors"))

    hf_cfg = soda_hier_config_from(handle.config.hier_model)
    cfg_dict = hf_cfg.to_dict()
    rope = cfg_dict.pop("rope_parameters", None)  # transformers-5 emits this; 4.x needs legacy keys
    if rope is not None:
        cfg_dict["rope_theta"] = rope.pop("rope_theta", 500000.0)
        cfg_dict["rope_scaling"] = rope or None
    cfg_dict["architectures"] = ["SodaHierForCausalLM"]
    cfg_dict["auto_map"] = {
        "AutoConfig": "configuration_soda_hier.SodaHierConfig",
        "AutoModelForCausalLM": "modeling_soda_hier.SodaHierForCausalLM",
    }
    with open(out / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)

    pkg = Path(__file__).parent
    for name in ("configuration_soda_hier.py", "modeling_soda_hier.py"):
        shutil.copy(pkg / name, out / name)
    AutoTokenizer.from_pretrained(TOKENIZER_ID).save_pretrained(out)

    _assert_loadable(handle)
    logger.info("%s: conversion verified at %s", run, handle.hf_out_dir)
    return handle


def _assert_loadable(handle: RunHandle) -> None:
    model, info = AutoModelForCausalLM.from_pretrained(
        handle.hf_out_dir, torch_dtype=torch.float32, trust_remote_code=True, output_loading_info=True
    )
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise AssertionError(f"{handle.run}: loading problems {problems}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s: reload OK, %d params", handle.run, n_params)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=hier_runs(), required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    convert_run(args.run, args.step)


if __name__ == "__main__":
    main()
