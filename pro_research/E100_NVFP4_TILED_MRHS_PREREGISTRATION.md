# E100-NVFP4-TILED-MRHS preregistration

Date: 2026-08-16
Branch: `pro-e100-batch`
Base result evidence: E100-NVFP4-SMEM-MRHS v4 smoke, which produced one exact `shared_up_nvfp4` N=4 record and then terminated technically before `shared_down`/`lm_head` because whole-row FP32 staging can exceed per-CTA shared-memory capacity.

## Why this is a new experiment

V4 staged `ROW_TILE * cols` dequantized FP32 weights at once. That is structurally valid for `cols=2688` at N=4 (43,008 dynamic shared-memory bytes), but it is not a general kernel for the larger shared-down input width. The technical failure therefore does **not** answer whether one packed NVFP4 stream can be decoded once and reused by multiple RHS values.

V5 changes only the staging schedule. It does not change arithmetic, quantization, model state, routing, or the reference baseline.

## Frozen mechanism

For each output-row tile and each 256-vector epoch:

1. one CTA decodes at most 256 packed `uchar4` vectors per row into transient FP32 shared memory;
2. each packed vector represents eight consecutive weights;
3. N independent 16-lane ERVF subgroups consume that same decoded tile, one subgroup per `(row,rhs)` pair;
4. every physical lane retains the same 16 virtual-reference accumulators for the entire K dimension;
5. tile epochs advance by exactly 256 packed vectors, so virtual reference tid `t` consumes `v=t, t+256, t+512, ...` in the same order as adopted V6 ERVF;
6. the width-16 reduction tree is reconstructed exactly as in the adopted single-RHS kernel.

`ROW_TILE * NRHS == 16`, keeping the CTA at 256 threads:

- N=4 -> ROW_TILE=4;
- N=8 -> ROW_TILE=2;
- N=16 -> ROW_TILE=1.

The fixed packed-vector tile is 256 vectors = 2048 FP32 weights per row. Maximum dynamic shared memory is therefore:

- N=4: 4 * 2048 * 4 = 32,768 B;
- N=8: 16,384 B;
- N=16: 8,192 B.

This cap is independent of matrix `cols`.

## Cases

Only the three NVFP4 common-weight families are in scope:

- `shared_up_nvfp4`;
- `shared_down_nvfp4`;
- `lm_head_nvfp4`.

Smoke runs N=4. Full runs N=4,8,16.

## Correctness gates

For every case and N:

- original production single-RHS == adopted V6 single-RHS bit-for-bit;
- adopted V6 == tiled-MRHS candidate bit-for-bit for every RHS/output element;
- repeated candidate call is bit-for-bit deterministic;
- all outputs finite;
- all three mandatory families must execute.

Any correctness failure closes this candidate immediately.

## Performance measurement

Frozen A/B order after both arms are warmed:

`REF_A -> CAND_A -> CAND_B -> REF_B`

Performance reference is N sequential calls through the **adopted V6** single-RHS dispatcher, never the legacy production GEMV.

Reference drift is `abs(A-B)/midpoint`.

## Frozen performance gates

Smoke is diagnostic and cannot create an E100 claim.

A full-run candidate requires at N=16:

- weighted speedup across the three registered NVFP4 families >= 2.00x;
- lm_head speedup >= 2.00x;
- shared_up speedup >= 1.20x;
- shared_down speedup >= 1.20x;
- no individual case below 1.20x;
- maximum reference drift <= 7%;
- all correctness gates pass.

These gates are frozen before any V5 target-GPU result.

## Claim boundary

This is a component test. Even a passing N=16 result is not a 100 tok/s full-model claim. It only licenses integration into a fixed-N graph-resident batch runtime, where end-to-end aggregate throughput must be measured independently.
