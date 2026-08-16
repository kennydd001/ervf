# E100-NVFP4-SMEM-MRHS preregistration — exact shared-decode weight reuse

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: the earlier MRHS32/MRHS256 N=4 smoke exists; **no target result exists for this new geometry**.

## Motivation from the already-observed smoke

The first two exact multi-RHS kernels established that common-weight arithmetic can remain bit-identical, but their N=4 performance was poor on the large NVFP4 matrices:

- MRHS32 weighted registered speedup: ~0.943x; LM-head ~0.654x.
- MRHS256 weighted registered speedup: ~0.933x; LM-head ~0.655x.

This is not reinterpreted as a pass. Those two geometries keep N RHS accumulators in the same physical worker (MRHS32) or fall back to one output row per 256-thread CTA without the adopted ERVF row packing (MRHS256). Both give up important occupancy/row-reuse properties of the V6 NVFP4 ERVF baseline.

The new hypothesis is narrower: **share the matrix bytes and decode across RHS without multiplying per-thread accumulator state by N.**

## Frozen kernel geometry

Only NVFP4 matrices are tested first because shared-up, shared-down and LM-head are the stable regressions that dominate the failed MRHS smoke.

For N in `{4,8,16}`:

- width = 16 physical lanes per `(output_row, rhs)` subgroup, exactly matching the adopted width-16 ERVF virtual-thread map;
- `ROW_TILE = 16 / N`, therefore one 256-thread CTA always contains `ROW_TILE * N` independent 16-lane subgroups;
- one subgroup owns one `(row,rhs)` result and keeps exactly 16 virtual-thread accumulators, independent of N;
- before any MAC, the CTA decodes each of its `ROW_TILE` NVFP4 rows **once** into transient dynamic shared-memory float weights;
- all N RHS subgroups for a row read those identical shared decoded weights;
- no RHS accumulator is shared with another RHS;
- the reference virtual tid assignment, MAC order and two-stage reduction tree are reproduced exactly for every RHS.

For `cols=2688` the decoded-weight shared-memory footprint is fixed and small enough for the target geometry:

- N=4, ROW_TILE=4: 43,008 B;
- N=8, ROW_TILE=2: 21,504 B;
- N=16, ROW_TILE=1: 10,752 B.

The total number of 256-thread CTAs across N sequential reference launches and one candidate launch is therefore of the same order; the candidate is not allowed to win merely by doing less arithmetic. It targets N-fold global matrix traffic/decode reuse.

## Exactness contract

For each real checkpoint case and deterministic activation batch:

1. production single-RHS output;
2. adopted V6 NVFP4 ERVF output;
3. shared-decode MRHS output;
4. deterministic repeat of shared-decode MRHS.

Every output float must satisfy:

`production == adopted == candidate == candidate_repeat` bit-for-bit.

The shared decode computes the same float expression as production before storing it:

`scale = e4m3_lut[group] * global_scale`

`weight = e2m1_lut[nibble] * scale`

Storing/loading this already-rounded float from shared memory must not alter its bits. The subsequent `fmaf(weight, x, acc)` order is unchanged.

## Frozen real-checkpoint families

All are mandatory:

- `shared_up_nvfp4`;
- `shared_down_nvfp4`;
- `lm_head_nvfp4`.

No synthetic substitute may rescue a missing family.

## Timing

CUDA-event device time, already-compiled modules, warm both arms once, then frozen `REF, CAND, CAND, REF` ordering.

Smoke:

- N=4 only;
- one deterministic correctness batch;
- short timing, diagnostic only.

Full:

- N=4,8,16;
- >=3 deterministic correctness batches per family/N;
- >=10 repeats and >=4 timing rounds per A/B/B/A arm.

Raw samples are retained.

## Gates

Correctness, mandatory at every N:

- all three families supported;
- production/adopted/candidate/repeat bit-identical;
- finite outputs;
- adopted NVFP4 ERVF reference exercised.

The primary performance arm is N=16. A full `smem_mrhs_candidate` requires:

- weighted registered NVFP4 aggregate speedup >= **2.0x**;
- LM-head speedup >= **2.0x**;
- no NVFP4 family below **1.20x**;
- reference A/B drift <= **7%** for every N=16 family.

N=4 and N=8 are scaling diagnostics and cannot rescue a failed N=16 primary gate.

## Claim boundary

This remains a component primitive. Even a 2–4x NVFP4 multi-RHS win is **not** an E100 runtime claim. Integration requires a fixed-N graph-resident full model with exact per-sequence causal parity and measured aggregate throughput. Component speedups are never multiplied into an end-to-end claim.
