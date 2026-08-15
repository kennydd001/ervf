# HET-NEXT L0 PH0X-R9 — direct no-FTZ PTX NVIDIA completion preregistration

Date: 2026-08-13. Exploratory NVIDIA repair only; no formal PH0 pass.

## Frozen rationale and evidence

- R7 result SHA `314e08fc907965cf13b2af110b6a45424a9ac75ec5ec429b8f7bc7bf99fdba53` remains formally negative (122/512 differences), with all non-numerical NVIDIA gates positive.
- Independent CPU replay proved the stored R7 NVIDIA SHA `6525b36b...` equals an exact PTX-FTZ emulator 512/512. Local CuPy compiler source SHA `8a8745a8...` appends `-ftz=true` after caller options.
- Direct-NVRTC R8 textual PTX has SHA `ec4789735f548123be0df3c2ff20c3e05c7b3741d9ed5f00b7b51eaeaa8ca7ae`, 133,404 bytes, zero `.ftz`, 256 `mul.f32`, 256 `fma.rn.f32`, 34 `add.rn.f32` and an empty compiler log.
- R8-R1 parser-correction JSON SHA `171650e58abef1dd9224e3d2a6db1a0b74f56c99e3a0bf5887299d6d2b3713a0` proves exactly three width-8 shuffle instructions with offsets 4/2/1 and encoded clamp/segment 6175 (`0x181f`).
- The bound Intel R3 arm is not rerun and remains 512/512 bitwise CPU-equal with clean lifecycle.

## One-attempt repair

R9 uses the exact R7 NVIDIA buffer, copy, launch, synchronization, release and comparison path, but intercepts its single `RawModule(code=...)` construction. The interceptor rejects any source/options/name-expression mismatch and loads the already frozen no-FTZ PTX by path. The original CuPy factory is restored on every path. No source compilation occurs.

The exact 24-row ledger is unchanged except row 0 is `ptx_load` binding PTX SHA/bytes instead of `compile`. All R7 structured failure evidence and lifecycle repairs remain. Before any payload/device action, R9 verifies R5/R6/R7 dependencies, R7 result, R8 diagnostic/PTX and R8-R1 correction hashes and predicates.

Positive requires NVIDIA 512/512 BF16 words bitwise equal to the unchanged strict CPU/Intel oracle SHA `e8a00c17...`, counters one, sentinel overwritten, exact identity, exact ledger and clean releases. Any negative/error exits nonzero. One new output directory, one attempt, no retry or retuning.

Claim boundary: this may establish only that CPU, Intel host-USM and NVIDIA direct-no-FTZ PTX reproduce one official real Q5 projection on one known natural activation. No full expert, MoE, layer, model, held-out/generalized quality, cohabitation, concurrency, timing, performance, deployment, novelty or breakthrough claim.
