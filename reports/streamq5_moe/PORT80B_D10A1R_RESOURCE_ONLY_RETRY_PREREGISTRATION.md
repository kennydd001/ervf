# PORT80B-D10A1-R — resource-only retry preregistration

## Scope and immutable base

D10A1-R is a resource-only retry of the unmodified D10A1 component executor.
It changes exactly one gate: available RAM required before host registration.
It does not change the CUDA sources, physical buffers, 499+13 transport,
routes, canaries, numerical oracles, controls, warm-ups, validation cases,
timing thresholds, page-read gates, post-registration/first-touch/emergency
stops, VRAM reserve, cleanup, claims, or fail-closed endurance behavior.

The inherited base is locked to:

- D10A1 preregistration SHA-256
  `ee71a17bf8889e009f8f692aa1dbafddcbd8691b68aa808ab09df0977d607b91`;
- D10A1 runner SHA-256
  `ffde9c13a3d6d19e3e1132369a4eb9a2e98a4e974bbece86ec224e2931f0ecfd`.

The prior D10A1 resource-stop JSON and report remain immutable and are bound by
SHA-256. D10A1-R writes only new `d10a1r_resource_only_retry` paths and refuses
overwrite.

## Independently audited start formula

The frozen component trace touches an exact union of 14,452 whole expert
records:

```
29,301,719,040 unique mmap bytes
+2,147,483,648 post-first-touch reserve
+1,073,741,824 explicit OS/background margin
=32,522,944,512 bytes start gate
```

Before compile and again before component execution, the runner independently
reconstructs the set union from the frozen 40 correctness cases, first 32
validation cases, 48 shared records, ten layer-0 reference records, and both
negative-control sources. It requires 14,404 routed-union records, 14,452
complete records, and the exact byte formula above.

The retry binds both available independent resource audits:

- `D10A1_FIRST_TOUCH_INDEPENDENT_SOURCE_AUDIT_2026-08-13.md`, SHA-256
  `869e56574082e96ce960662dda3dd7e542cd814fb467bf5051831a6efefac081`;
- `port80b_d9_capacity_aware_bank_bridge_independent_verification.json`,
  SHA-256
  `629593339d9e39d7ce12d8e85277d6bb9f37ee7316afb426b71529a2c37a6747`.

The first establishes the exact frozen-trace union and recommended gate. The
second preserves the empirical caveat that registration alone did not first
touch the mapped bank, while execution did and clean CUDA unregister did not
imply prompt OS page reclamation.

## Gates that do not change

- Immediately after registering exactly 48 × 499 records: at least 2 GiB
  available RAM, otherwise synchronize/unregister/cleanup and fail.
- After eight frozen warm-ups/first touch: at least 2 GiB available RAM.
- During measured validation: emergency abort below 1.5 GiB.
- Device allocation remains exactly 4,521,569,280 bytes with at least 512 MiB
  free VRAM reserve.
- All three radix-32 canaries, exhaustive 24,576-ID injectivity/roundtrip and
  every-layer 498↔499 boundary checks remain mandatory.
- All 40 correctness cases, raw intended/actual canary arrays, full routed Q5
  bitexact oracle, wrong-layer/wrong-expert differentiated numerical controls,
  attention/GDN/shared/dense oracles, 8 warm-ups and 32 measured validation
  cases remain unchanged.
- Inclusive wall p95 ≤150 ms and p99 ≤200 ms, page reads ≤2,048/s, validation
  RAM loss ≤1 GiB, finite outputs, no CUDA error, and clean unregister of all
  48 ranges remain mandatory.

No component run is authorized by this document or its compile phase. A GPU run
requires a separate explicit go. Endurance remains closed even after compile;
the inherited 10,000-step executor is deliberately fail-closed pending a clean
component pass, its own first-touch calculation, and separate authorization.

## New evidence paths

- runner: `scripts/streamq5_moe/run_port80b_d10a1r_resource_only_retry.py`;
- compile JSON: `reports/streamq5_moe/port80b_d10a1r_resource_only_retry_compile.json`;
- compile report: `reports/streamq5_moe/PORT80B_D10A1R_RESOURCE_ONLY_RETRY_COMPILE_REPORT_2026-08-13.md`;
- component JSON: `reports/streamq5_moe/port80b_d10a1r_resource_only_retry.json`;
- component report: `reports/streamq5_moe/PORT80B_D10A1R_RESOURCE_ONLY_RETRY_REPORT_2026-08-13.md`.

## Claim boundary

A future pass would remain only a synthetic shape-informed physical component/
composition stress result on `p4d_shaped_synthetic_proxy` routes. D10A1-R is
not an exact official Qwen3-Next shell, real checkpoint, natural route stream,
model-quality result, production throughput result, or endurance result. The
lower start gate is valid only for this frozen component trace and cannot be
reused for the 10,000-step route union.
