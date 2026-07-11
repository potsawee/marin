# Multi-GPU smoke "passed" with train/loss 0.0 — a stale-resume no-op

**Date:** 2026-07-08 (validating multi-GPU before scaling the campaign).

## Question

A 2-GPU smoke of both arms reported `RUNG ... PASS: final train loss 0.000`. A
real cross-entropy never hits exactly 0.0 — was multi-GPU corrupting training?

## Conclusion

**Not a multi-GPU bug.** The smoke used a fixed run id (`smoke-armf-tiny`) with a
persistent checkpointer. The 2-GPU run found the *previous* single-GPU run's
`step-99` checkpoint, saw it was already at the target step count, and ran **zero
training steps**:

```
Loading checkpoint from .../smoke-armf-tiny/ckpt/.../step-99
Training already complete at step 100 (target: 100). Running final hooks only.
```

`train/loss: 0.0` was an unpopulated metric (no step ran). The assertion
`loss < 9.0` passed it falsely. Data-parallel itself is correct: after wiping the
stale checkpoints, the clean 2-GPU rerun reproduced the single-GPU losses exactly
(armf-tiny 11.9→8.07, overfit 11.8→3.93), and global batch is device-count-
independent, so multi-GPU is pure (if sub-linear, see `mfu-throughput.md`)
speedup at fixed isoflop budget.

## Fixes (committed)

- **Config-hash run names** (`exp_isoflop_headline.experiment_signature`): the
  Marin-canonical `fingerprint_hash(canonical_json(...))` over the experiment
  (model/data/optimizer/loss recipe/global batch/steps/seed) but NOT infra
  (device count, mesh, paths). Same experiment → same hash → correct resume
  across GPU counts; different config → different hash → no stale-checkpoint
  collision. Verified: `p1-hier` and `p1b-hier` share a solver spec but get
  distinct hashes.
- **Hardened smoke** (`exp_smoke.py`): each rung wipes its store before training,
  and asserts a plausible loss floor (`0.5 < loss`) so a no-op resume fails
  loudly instead of passing.
