# PH1 Intel execution R6P1 — independent frozen source audit

Date: 2026-08-14  
Scope: read-only source/method audit; the preflight, payload, compiler, and device were not executed.

## Verdict

**GO for exactly one execution of the frozen, device-free R6P1 static preflight.**

This verdict does **not** authorize reading the PH1 payload, invoking OpenCL, compiling a kernel, or running the physical Intel execution. Those stages remain closed unless this exact preflight emits its own complete PASS result and that result is separately reviewed.

## Frozen identity and clean-state checks

The observed SHA-256 values match the handoff:

- `scripts/streamq5_moe/preflight_het_next_l0_ph1_intel_execution_r6p1.py`: `c17b5a8e934097bd8939d21af2ab6a6b585adfc9367f15bf468c64dd2b4f7590`
- `reports/streamq5_moe/HET_NEXT_L0_PH1_INTEL_EXECUTION_R6P1_PREREGISTRATION_2026-08-14.md`: `93db7ae2c5a412342612da392314547faf6f7d0604c6500b8f61ed9458ac65de`
- `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r6p1_lock.json`: `9b438aad23c0a2edcf26585e90812260185b5541d8102fe0c7d1377e0bf3f391`

The R6 physical output directory and the R6P1 preflight-result file are absent. The lock is closed (`execution_open=false`, `audit_token=PENDING`) and binds the immutable R6P failure result, its independent diagnosis, the R6P source/audit, and the earlier R6 audit/crash diagnosis.

## Targeted repair audit

All three preregistered R6P failures are closed in the frozen source:

1. **AST device-loader detection is non-vacuous and avoids textual self-match.** `loader_call_names()` walks `ast.Call` nodes only. The positive sentinel detects a real `WinDLL(...)` call, while the negative sentinel accepts the same spelling inside a string literal. The production gate rejects loader calls and forbidden imports on the actual static-preflight/common/verifier trees. It therefore no longer fails merely because a blacklist token occurs in source text.

2. **The codec fixture uses the actual BF16-scale quantization result.** The frozen expected vector is `[-15,-7,-4,0,4,7,11,15]`, matching the defined BF16-scale path rather than the rejected idealized `-8/+8` values.

3. **The full-shape verifier baseline is internally bound rather than zero-weight-vacuous.** Each synthetic wire record is independently parsed (header, CRC, 5-byte packed fields, forbidden field 31, BF16 scales) into the weight tensor. The source asserts q=1 at slot 0 of every 8-wide group and q=0 elsewhere, scale exactly BF16 1.0. Those decoded tensors feed both fixture construction and the independent verifier. The input is deliberately zero, so zero outputs are mathematically expected despite nonzero bound weights.

The schema check uses production shapes `(512,2048)`, `(512,2048)`, `(2048,512)`, exact output/counter byte lengths, production `BUFF`, `ARGS`, and `LAUNCH`, and explicitly compares `verify.BUFF` to the saved production table rather than to itself.

## Baseline and mutation non-vacuity

The source requires the complete baseline check map returned by the real independent `verify.verify_dict()` to be true before mutation testing. It records both the map and `baseline_false_names`.

Exactly 28 named, one-at-a-time production-result mutations are constructed and passed back through the same independent verifier using `zip(..., strict=True)`. Every one must make the complete check map non-all-true; the preflight requires the rejected-name list to equal the full frozen 28-name list. The mutations cover status/error returns, ownership missing/duplicate/return/pending/pointer, identity, controls, output, pointer alias/alignment/USM metadata, argument pointer, launch geometry/event, read/release order, ownership/release codes, cleanup, provenance, resource summary/order/peak, forbidden API, and stage hash.

This is non-vacuous: the baseline must first pass, production-sized weights/counters are used, and a mutation is counted only after an actual verifier rejection. The result stores the baseline map, false-name list, mutation-name list, and rejected-name list. Per-mutation false-check maps are not retained, but that is not a preregistered gate and does not weaken the Boolean rejection claim.

## Remaining boundary

No deterministic execution blocker was found in the frozen R6P1 source. The full-shape integer oracle is CPU-heavy because it is replayed for the baseline and all 28 mutations; that is a runtime-cost caveat, not a scientific or authorization blocker.

Authorized next action: run **only** the exact SHA-bound R6P1 no-device static preflight once. A physical execution remains unauthorized until the resulting preflight artifact independently demonstrates every gate true.
