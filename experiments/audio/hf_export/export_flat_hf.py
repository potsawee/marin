# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export an Arm-F (flat) run to a HuggingFace Qwen3ForCausalLM checkpoint.

Output mirrors the old SODA layout: <run_root>/hf/step-N/{config.json,
model.safetensors, tokenizer files}. Two consumer constraints drive the
post-processing:

- blueberry-eval and third-party loaders may run transformers 4.x, which does
  not read the transformers-5 ``rope_parameters`` config key — the config is
  rewritten to the legacy ``rope_theta``/``rope_scaling`` keys (5.x accepts
  them too).
- The exported tokenizer must auto-prepend BOS (the paired text evals assume
  it); the hub tokenizer already does this via its post_processor, asserted
  after export.

Usage (CPU, run under the marin venv with env.sh sourced):
    python -m experiments.audio.hf_export.export_flat_hf --run p1-flat [--step N]
    python -m experiments.audio.hf_export.export_flat_hf --all-flat
"""

import argparse
import json
import logging

import torch
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.main.export_lm_to_hf import main as export_main
from levanter.trainer import TrainerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.audio.audio_vocab import BOS_ID, FULL_VOCAB, TOKENIZER_ID
from experiments.audio.hf_export.run_registry import RunHandle, flat_runs, resolve

logger = logging.getLogger(__name__)


def _patch_rope_to_legacy_keys(config_path: str) -> dict:
    """Rewrite transformers-5 ``rope_parameters`` to legacy rope keys, in place."""
    with open(config_path) as f:
        cfg = json.load(f)
    rope = cfg.pop("rope_parameters", None)
    if rope is not None:
        cfg["rope_theta"] = rope.pop("rope_theta", 500000.0)
        cfg["rope_scaling"] = rope or None
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return cfg


def _assert_export(handle: RunHandle, cfg: dict) -> None:
    model = handle.config.flat_model
    expect = {
        "vocab_size": FULL_VOCAB,
        "hidden_size": model.hidden_dim,
        "intermediate_size": model.intermediate_dim,
        "num_hidden_layers": model.num_layers,
        "num_attention_heads": model.num_heads,
        "num_key_value_heads": model.num_kv_heads,
        "head_dim": model.hidden_dim // model.num_heads,
        "max_position_embeddings": model.max_seq_len,
        "tie_word_embeddings": False,
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
    }
    for key, want in expect.items():
        got = cfg.get(key)
        if got != want:
            raise AssertionError(f"{handle.run}: config.json {key}={got!r}, expected {want!r}")
    scaling = cfg.get("rope_scaling") or {}
    if cfg.get("rope_theta") != 500000.0 or scaling.get("rope_type", scaling.get("type")) != "llama3":
        raise AssertionError(f"{handle.run}: rope not llama3/500000: theta={cfg.get('rope_theta')} scaling={scaling}")


def _assert_loadable(handle: RunHandle) -> None:
    """Reload with transformers and assert weights + tokenizer are exactly right."""
    tok = AutoTokenizer.from_pretrained(handle.hf_out_dir)
    if len(tok) != FULL_VOCAB:
        raise AssertionError(f"{handle.run}: exported tokenizer len {len(tok)} != {FULL_VOCAB}")
    if tok("x").input_ids[0] != BOS_ID:
        raise AssertionError(f"{handle.run}: exported tokenizer does not auto-prepend BOS {BOS_ID}")

    model, info = AutoModelForCausalLM.from_pretrained(
        handle.hf_out_dir, torch_dtype=torch.float32, output_loading_info=True
    )
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise AssertionError(f"{handle.run}: loading problems {problems}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s: reload OK, %d params", handle.run, n_params)


def export_run(run: str, step: int | None = None) -> RunHandle:
    handle = resolve(run, step)
    if handle.arm != "flat":
        raise ValueError(f"{run} is arm={handle.arm}; use convert_hier_to_hf.py for hier runs")
    logger.info("exporting %s (%s step %d) -> %s", run, handle.run_id, handle.step, handle.hf_out_dir)
    export_main(
        ConvertLmConfig(
            trainer=TrainerConfig(require_accelerator=False),
            checkpoint_path=handle.step_dir,
            output_dir=handle.hf_out_dir,
            model=handle.config.flat_model,
            tokenizer=TOKENIZER_ID,
            checkpoint_subpath="model",
            save_tokenizer=True,
            use_cpu=True,
        )
    )
    cfg = _patch_rope_to_legacy_keys(f"{handle.hf_out_dir}/config.json")
    _assert_export(handle, cfg)
    _assert_loadable(handle)
    logger.info("%s: export verified at %s", run, handle.hf_out_dir)
    return handle


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", choices=flat_runs())
    group.add_argument("--all-flat", action="store_true")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    args = parser.parse_args()

    runs = flat_runs() if args.all_flat else [args.run]
    for run in runs:
        export_run(run, args.step)


if __name__ == "__main__":
    main()
