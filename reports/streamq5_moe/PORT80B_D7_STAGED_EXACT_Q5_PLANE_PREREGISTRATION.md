# PORT80B-D7 — cp.async staged exact Q5 plane preregistration

**Frozen before physical execution:** 2026-08-12

D6 proved exact direct host Q5 execution but failed latency because row-granular
remote loads reached 76.421 ms p50. D7 composes the two already locked passing
components without retuning: D5's selected 1024-block full-record `cp.async`
stream into a 973,209,600-byte HBM work buffer, followed serially by the exact
resident width-8 Q5 gate/up → canonical BF16 SwiGLU → down plane.

## Protocol

- same immutable bank, 307-expert/layer registered prefix and deterministic
  routes as D5/D6;
- one complete sample stages all 480 records and computes ten experts across all
  48 layers; no CPU bounce copy, dequantized weight matrix or cache assumption;
- full candidate outputs (1,474,560 float32 values) must be bitwise equal to the
  resident oracle, with equal SHA-256;
- fixed D5 schedule: 1024 blocks, 256 threads, 4-KiB SMEM tiles;
- fixed N4B-R schedule: ERVF width 8;
- 5 warm-ups, 24 validation samples, open test at validation p50 <=65 ms;
- 120 once-only test samples, no tuning or overlap after validation.

## Gates

Primary pass: exact outputs/digests, 120 finite test samples, expert-plane p95
<=65 ms, conservative payload rate >=15 GB/s, expert p95 + frozen 28.077227-ms
dense-shell p95 <=100 ms, clean 48-range registration/unregister and no error.

Strong pass: expert p95 <=55 ms and projected total <=90 ms.

## Claim boundary

This is a physically timed exact synthetic Q5 expert plane on a 60%-registered
bank. It still uses a 973-MB HBM work buffer and has no copy/compute overlap. It
is not a full-bank, real-checkpoint, natural-routing, quality, physical-shell,
end-to-end tokens/s or endurance result.
