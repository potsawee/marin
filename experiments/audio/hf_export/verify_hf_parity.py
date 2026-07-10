# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity check: an exported HF checkpoint vs the JAX training model.

Computes teacher-forced per-type NLL (the eval_audio_nll bucket schema) on the
same dev-clean docs with both stacks and compares per-bucket means. The JAX
hier path runs with z_loss_weight=0 because the training loss carries a
1e-4·logZ² penalty the HF logits path does not.

Usage (either stack alone is meaningless — this needs GPU or patience):
    python -m experiments.audio.hf_export.verify_hf_parity --run p1-flat --limit 16
"""

import argparse
import dataclasses
import logging

import numpy as np

from experiments.audio.eval_audio_nll import (
    DEFAULT_EVAL_PARQUET,
    _bucket_totals,
    eval_flat,
    eval_hier,
    load_eval_docs,
)
from experiments.audio.hf_export.run_registry import ALL_RUNS, resolve

logger = logging.getLogger(__name__)

# Cross-framework fp32 noise: JAX's blocked fused-CE logsumexp vs torch's
# dense log_softmax differ by up to ~1e-3 nats/token on the 144,644-way
# softmax (observed: semantic agrees to 1e-7, small-count buckets drift a few
# 1e-4). Downstream metrics resolve nothing below ~1e-2, so 1e-3 is strict.
TOLERANCE_NATS = 1e-3


def torch_bucket_totals(hf_dir: str, docs) -> dict[str, tuple[float, int]]:
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32, trust_remote_code=True)
        .to(device)
        .eval()
    )
    totals: dict[str, tuple[float, int]] = {}
    with torch.no_grad():
        for flat, _, _ in docs:
            ids = torch.tensor(np.asarray(flat), dtype=torch.long, device=device)[None]
            logits = model(ids).logits[0].float()
            logp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = ids[0, 1:]
            ce = -logp.gather(1, tgt[:, None])[:, 0]
            buckets = _bucket_totals(
                ce.cpu().numpy(), np.ones(len(ce), dtype=np.float32), tgt.cpu().numpy().astype(np.int64)
            )
            for name, (s, n) in buckets.items():
                s0, n0 = totals.get(name, (0.0, 0))
                totals[name] = (s0 + s, n0 + n)
    return totals


def jax_bucket_totals(handle, docs) -> dict[str, tuple[float, int]]:
    if handle.arm == "flat":
        return eval_flat(handle.step_dir, handle.config.flat_model, docs)
    cfg = dataclasses.replace(handle.config.hier_model, z_loss_weight=0.0)
    return eval_hier(handle.step_dir, cfg, docs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=sorted(ALL_RUNS), required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--hf", default=None, help="HF dir (default: the run's hf/step-N export)")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--eval-parquet", default=DEFAULT_EVAL_PARQUET)
    args = parser.parse_args()

    handle = resolve(args.run, args.step)
    hf_dir = args.hf or handle.hf_out_dir
    docs = load_eval_docs(args.eval_parquet, limit=args.limit)

    hf_totals = torch_bucket_totals(hf_dir, docs)
    jx_totals = jax_bucket_totals(handle, docs)

    print(f"{'bucket':<12} {'n':>7} {'HF mean':>12} {'JAX mean':>12} {'delta':>12}")
    failed = []
    for name in sorted(set(hf_totals) | set(jx_totals)):
        hs, hn = hf_totals.get(name, (0.0, 0))
        js, jn = jx_totals.get(name, (0.0, 0))
        if hn != jn:
            failed.append(f"{name}: count mismatch HF={hn} JAX={jn}")
            continue
        if hn == 0:
            continue
        delta = hs / hn - js / jn
        print(f"{name:<12} {hn:>7} {hs / hn:>12.6f} {js / jn:>12.6f} {delta:>12.2e}")
        if abs(delta) > TOLERANCE_NATS:
            failed.append(f"{name}: |delta| {abs(delta):.2e} > {TOLERANCE_NATS}")
    if failed:
        raise SystemExit("PARITY FAILED:\n  " + "\n  ".join(failed))
    print(f"PARITY OK: {args.run} @ {hf_dir}")


if __name__ == "__main__":
    main()
