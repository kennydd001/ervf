# PORT80B-D10B-R — held-out 10,000-step endurance provenance revision

Date: 2026-08-13

The first D10B CPU-only preflight failed before opening execution because its
inherited conv-unit check compared the D10A2-R2 unit result's runner and
preregistration hashes with the new D10B files rather than with the immutable
D10A2-R2 files. The failed preflight and its sources remain immutable:

- D10B runner SHA-256:
  `4f8226d82d7d804195a9728bc9852cc9b75fa33ec6d8481e86d94ae90ff3cb68`;
- D10B preregistration SHA-256:
  `8d171ac876d03681d35de9155b100ec01e3345588ca31eb926e9acddaa59b977`;
- failed preflight JSON SHA-256:
  `91bb855940f0d39f241c29159ae39c011c46e4e9a3297d50bdb696e90fde985e`;
- failed preflight report SHA-256:
  `c0e9fcbcc9e1010307bffe23334b9a829cf7d5f7310a55092a0bfb1eb2c4dd21`.

That preflight records every physical action false and is not an endurance
result.

D10B-R permits exactly one provenance-plumbing correction: validate the frozen
conv-unit result's `runner_sha256`, `preregistration_sha256` and
`unit_test_sha256` against the immutable D10A2-R2 lock set, rather than against
D10B-R's own runner/preregistration. No route, GPU source, numerical operation,
case order/count, digest cadence, resource gate, timing threshold, drift gate,
telemetry field, cleanup rule or claim boundary may change.

The complete execution protocol and gates are inherited byte-for-byte from
`PORT80B_D10B_HELDOUT_10000_ENDURANCE_PREREGISTRATION.md`: exact held-out route
SHA `85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20`,
eight warm-ups, exactly 10,000 measured steps, p95 ≤150 ms, p99 ≤200 ms,
last/first-1,000 p95 ratio ≤1.20, exactly 101 digest checkpoints and 909 array
records, full per-step state checks, raw latency/telemetry, 52,652,163,072-byte
start-RAM gate, paging/RAM/VRAM emergency gates and exact 48+48 lifecycle rows.

CPU preflight is authorized. It must record no CUDA initialization, NVRTC
compile, host registration, large allocation, kernel launch or bank scan. A
separate explicit GPU go remains mandatory.

Evidence paths:

- runner: `scripts/streamq5_moe/run_port80b_d10br_heldout_10000_endurance_revision.py`;
- preflight JSON:
  `reports/streamq5_moe/port80b_d10br_heldout_10000_endurance_revision_preflight.json`;
- preflight report:
  `reports/streamq5_moe/PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_PREFLIGHT_REPORT_2026-08-13.md`;
- raw result:
  `reports/streamq5_moe/port80b_d10br_heldout_10000_endurance_revision.json`;
- report:
  `reports/streamq5_moe/PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_REPORT_2026-08-13.md`.

Claim boundary remains synthetic shape-informed endurance on P4D-shaped proxy
routes and uniform Q5 payloads only; not checkpoint, natural routing, quality,
production throughput or breakthrough evidence.

