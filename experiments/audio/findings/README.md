# Findings

Self-contained write-ups of investigations run during this project — each one
states the question, the method/data, the conclusion, and the action taken.
Committed (unlike the transient `*.local.md` trackers) because they're part of
the engineering record and feed the eventual writeup.

- [`mfu-throughput.md`](mfu-throughput.md) — why P1 runs at ~6% MFU (~17h/run),
  and the campaign-scheduling consequence (1 GPU per run, not 4).
- [`checkpoint-resume-noop.md`](checkpoint-resume-noop.md) — the multi-GPU smoke
  that "passed" with train/loss 0.0 was a stale-checkpoint resume no-op; led to
  config-hash run names + a hardened smoke assertion.
