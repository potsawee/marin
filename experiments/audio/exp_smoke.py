# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""P0 smoke rungs for the audio-LM comparison (see SMOKE_LADDER.local.md).

Each rung is a tiny in-process run against the mini-smoke caches, sized to
finish in minutes on one GPU (`floors` runs on CPU):

    uv run python experiments/audio/exp_smoke.py --rung armf-tiny
    uv run python experiments/audio/exp_smoke.py --rung overfit
    uv run python experiments/audio/exp_smoke.py --rung depth0
    uv run python experiments/audio/exp_smoke.py --rung floors

PASS criteria are asserted, so a failing rung exits nonzero.
"""

import argparse
import dataclasses
import json
import logging
import os
import shutil

import haliax as hax
import jax.random as jrandom
import jmp
import numpy as np
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer import TrainerConfig

from experiments.audio.data import AudioStepDataset, AudioStepExample
from experiments.audio.eval_audio_nll import DEFAULT_EVAL_PARQUET, eval_flat, eval_hier, load_eval_docs, summarize
from experiments.audio.model_hier import AudioHierConfig, AudioHierModel
from experiments.audio.train_audio_lm import AudioTrainConfig, main

logger = logging.getLogger(__name__)

SODA_ROOT = "/nlp/scr/potsawee/workspace/soda-extension"
MINI_DATA_ROOT = f"{SODA_ROOT}/data/mini-smoke/out"
MINI_SOURCES = {"yodas": 0.3, "emilia_yodas": 0.4, "emilia": 0.3}
SMOKE_STORE = f"{SODA_ROOT}/marin-store/smoke"

TINY_FLAT = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=128,
    intermediate_dim=512,
    num_layers=2,
    num_heads=1,
    num_kv_heads=1,
    rope=Llama3RotaryEmbeddingsConfig(),
    tie_word_embeddings=False,
)

TINY_HIER = AudioHierConfig(
    max_steps=1024,
    hidden_dim=128,
    intermediate_dim=512,
    num_layers=2,
    num_heads=1,
    num_kv_heads=1,
    depth_hidden_dim=64,
    depth_intermediate_dim=256,
    depth_layers=2,
    depth_heads=1,
    depth_kv_heads=1,
    rope=Llama3RotaryEmbeddingsConfig(),
)


def _fresh(run_id: str) -> None:
    """Wipe a run's store dir so a smoke always trains from scratch (no stale resume)."""
    shutil.rmtree(f"{SMOKE_STORE}/{run_id}", ignore_errors=True)


def _trainer(run_id: str, steps: int, batch: int) -> TrainerConfig:
    return TrainerConfig(
        id=run_id,
        seed=0,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        tracker=JsonFileTrackerConfig(output_path=f"{SMOKE_STORE}/{run_id}"),
        train_batch_size=batch,
        num_train_steps=steps,
        steps_per_eval=10_000_000,  # no in-train eval during smokes
        checkpointer=CheckpointerConfig(base_path=f"{SMOKE_STORE}/{run_id}/ckpt"),
    )


def _optimizer(lr: float) -> CautiousConfig:
    return CautiousConfig(
        learning_rate=lr,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=0.1,
        beta1=0.95,
        beta2=0.98,
        epsilon=1e-15,
        max_grad_norm=1,
        adamc_weight_decay=True,
        lr_schedule="linear",
        decay=0.2,
    )


def _final_train_loss(run_id: str) -> float:
    path = f"{SMOKE_STORE}/{run_id}/eval_results.json"
    with open(path) as f:
        metrics = json.load(f)
    assert "train/loss" in metrics, f"no train/loss in {path}: keys={sorted(metrics)[:10]}"
    return float(metrics["train/loss"])


def rung_armf_tiny() -> None:
    """The shared main trains the FLATTENED arm end-to-end and the loss moves."""
    run_id = "smoke-armf-tiny"
    _fresh(run_id)
    cfg = AudioTrainConfig(
        arm="flat",
        data_root=MINI_DATA_ROOT,
        sources=MINI_SOURCES,
        trainer=_trainer(run_id, steps=100, batch=8),
        optimizer=_optimizer(3e-3),
        flat_model=TINY_FLAT,
        seq_len=4096,
    )
    main(cfg)
    final = _final_train_loss(run_id)
    assert 0.5 < final < 9.0, f"flat tiny final loss {final:.3f} implausible (expect ~8; 0.0 => no-op/resume)"
    print(f"RUNG armf-tiny PASS: final train loss {final:.3f} (init ~11.9)")


def rung_overfit() -> None:
    """Arm H trains end-to-end on the mini caches and the joint loss drops sharply."""
    run_id = "smoke-armh-overfit"
    _fresh(run_id)
    cfg = AudioTrainConfig(
        arm="hier",
        data_root=MINI_DATA_ROOT,
        sources=MINI_SOURCES,
        trainer=_trainer(run_id, steps=300, batch=8),
        optimizer=_optimizer(3e-3),
        hier_model=TINY_HIER,
    )
    main(cfg)
    final = _final_train_loss(run_id)
    assert 0.5 < final < 7.0, f"hier tiny final loss {final:.3f} implausible (expect ~4; 0.0 => no-op/resume)"
    print(f"RUNG overfit PASS: final joint loss {final:.3f} (init ~11.8)")


def rung_depth0() -> None:
    """Monotonicity contract: removing depth capacity must not IMPROVE the joint loss.

    Trains the tiny hier model with a crippled depth stack and checks its loss is
    >= the full tiny model's from rung_overfit (same data, steps, seed).
    """
    run_id = "smoke-armh-depth0"
    _fresh(run_id)
    crippled = dataclasses.replace(TINY_HIER, depth_layers=0)
    cfg = AudioTrainConfig(
        arm="hier",
        data_root=MINI_DATA_ROOT,
        sources=MINI_SOURCES,
        trainer=_trainer(run_id, steps=300, batch=8),
        optimizer=_optimizer(3e-3),
        hier_model=crippled,
    )
    main(cfg)
    crippled_loss = _final_train_loss(run_id)
    full_loss = _final_train_loss("smoke-armh-overfit")
    assert (
        crippled_loss >= full_loss - 0.05
    ), f"depth-0 model beat the full model ({crippled_loss:.3f} < {full_loss:.3f}) - conditioning bug?"
    print(f"RUNG depth0 PASS: crippled {crippled_loss:.3f} >= full {full_loss:.3f}")


def rung_floors() -> None:
    """Random-init CE floors match the head supports on REAL cached windows."""
    ds = AudioStepDataset.load(f"{MINI_DATA_ROOT}/arm_h/yodas/train").as_sync_dataset()
    batch = [ds[i] for i in range(4)]
    example = AudioStepExample(
        codes=hax.stack("batch", [b.codes for b in batch]),
        seg_ids=hax.stack("batch", [b.seg_ids for b in batch]),
    )
    model = AudioHierModel.init(TINY_HIER, key=jrandom.PRNGKey(0))
    parts = model.per_type_losses(example)
    w = parts["w_backbone"]
    bb = float((parts["ce_backbone"] * w).sum() / w.sum())
    wd = parts["w_depth"][..., None]
    dep = float((parts["ce_depth"] * wd).sum() / (wd.sum() * 7))
    # Init CE sits ~logit_std^2/2 above ln(V); the unified head's ~1/sqrt(d) init
    # gives logit std ~1 -> ~+0.5 excess. A wrong support would differ by nats.
    assert -0.2 < bb - np.log(130308) < 1.0, f"backbone floor {bb:.2f} not ~ln(130308)=11.78"
    assert -0.2 < dep - np.log(2048) < 0.5, f"depth floor {dep:.2f} not ~ln(2048)=7.62"
    print(f"RUNG floors PASS: backbone {bb:.3f} (~11.78), depth {dep:.3f} (~7.62)")


def rung_evaldet() -> None:
    """Rung 6: the evaluator loads real checkpoints and is bit-deterministic."""
    docs = load_eval_docs(DEFAULT_EVAL_PARQUET, limit=64)
    ckpt_h = f"{SMOKE_STORE}/smoke-armh-overfit/ckpt/smoke-armh-overfit/step-299"
    ckpt_f = f"{SMOKE_STORE}/smoke-armf-tiny/ckpt/smoke-armf-tiny/step-99"
    for name, fn in (
        ("hier", lambda: eval_hier(ckpt_h, TINY_HIER, docs)),
        ("flat", lambda: eval_flat(ckpt_f, TINY_FLAT, docs)),
    ):
        a = summarize(fn(), docs)
        b = summarize(fn(), docs)
        assert a == b, f"{name} eval not deterministic"
        assert np.isfinite(a["nll/total"]) and a["bits_per_audio_second"] > 0
        print(f"  {name}: nll/total={a['nll/total']:.1f} bits/audio-s={a['bits_per_audio_second']:.2f} (2x identical)")
    print("RUNG evaldet PASS")


RUNGS = {
    "armf-tiny": rung_armf_tiny,
    "overfit": rung_overfit,
    "depth0": rung_depth0,
    "floors": rung_floors,
    "evaldet": rung_evaldet,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=tuple(RUNGS), required=True)
    args = parser.parse_args()
    os.makedirs(SMOKE_STORE, exist_ok=True)
    RUNGS[args.rung]()
