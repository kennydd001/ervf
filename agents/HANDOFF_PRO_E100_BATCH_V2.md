# Handoff V2 — PRO E100 exact batch / common-weight line

Active branch: **`pro-e100-batch`**.
Do not write `main`. Claude's V12/H-SCALE/Mamba campaign is a separate line and may have newer local changes not yet on remote. Do not rebase this branch onto unverified local output while either campaign is in flight.

## Target semantics

There are two separate future claims:

1. **aggregate E100**: independent exact autoregressive sequences collectively generate >=100 verified tokens/s;
2. **single-stream E100**: one exact autoregressive sequence generates >=100 verified tokens/s.

This branch currently attacks the first claim. Never label aggregate throughput as single-stream throughput.

A full aggregate E100 claim eventually requires exact per-sequence greedy parity, >=100 aggregate tok/s over >=1000 timed generated tokens, stable controls, reported per-sequence latency/gaps, and an independent verifier recomputing the claim from raw evidence.

## Corrected inherited facts

The old N=8 result of 7.3896 aggregate tok/s is superseded by `diag_n8_timing_reconcile.json`. The identical N=8 loop measured ~30.30-30.48 aggregate tok/s with CUDA events and synchronized wall timing agreeing within ~0.6%. N=8 is essentially flat versus solo, not a 4x physical collapse.

Device-cache hit rate also improves with N=8 (77.08% versus 69.71% solo), so cache thrash is not the explanation for absent scaling.

Measured route union across 13,800 layer/step observations:

- N=2: 11.584 / 12;
- N=4: 21.665 / 24;
- N=8: 38.903 / 48;
- N=16: 63.898 / 96.

Overlap is modest at N=4 and material at N=16. Route-union sharing alone is not enough for E100.

## Primary new mechanism: exact multi-RHS common-weight reuse

The old batch design treated attention/Mamba/LM-head as non-shareable because they have no expert identity to deduplicate. That confuses routing identity with matrix traffic.

Every active sequence uses the same dense/common matrix. N sequential GEMVs therefore stream that matrix N times. MRHS instead loads/dequantizes one weight scalar once, applies it to N independent RHS activations, keeps N independent accumulators, and preserves each RHS's production virtual-thread assignment, FMA order and reduction tree.

No two sequences share an accumulator or floating-point reduction. This is exact weight-traffic reuse, not approximation.

## Critical pre-run baseline audit

V1/V2 MRHS code timed BF16/F32/FP8 against the original 256-thread production GEMV. That is exact but stale as a performance baseline because V6 already adopted selective DenseERVF.

Only **V3** result files are eligible for decisions:

- production single-RHS is retained for correctness;
- adopted V6 selective single-RHS is the performance REF;
- every batch proves `production == adopted == candidate == candidate_repeat` bitwise;
- BF16 adopted ERVF shapes are `(4096,2688)` and `(2688,4096)`;
- FP8 adopted ERVF shapes are `(10304,2688)` and `(2688,4096)`;
- small K/V and F32 router remain production;
- NVFP4 uses adopted FusedNVFP4 ERVF.

The numerical gates were not relaxed when this correction was made.

## Primitive A — MRHS32

Files:

- `E100_MRHS_PREREGISTRATION.md`
- `E100_MRHS_V3_ADOPTED_BASELINE.md`
- `mrhs_exact_kernels.py`
- `e100_adopted_baseline.py`
- `e100_mrhs_adopted_bench.py`
- `e100_mrhs_v3.py`
- `verify_e100_mrhs_v3.py`
- `RUN_E100_MRHS.ps1`

Geometry: 32 physical lanes/output row; each lane emulates 8 production virtual tids. Fixed compile-time N=2/4/8.

V3 also measures Mamba output; it cannot substitute for any missing original frozen support family.

Primary N=4 full gates remain:

- exact production/adopted/candidate chain;
- >=6/7 frozen families with LM head and Mamba-in mandatory;
- weighted registered speedup >=1.75x versus adopted V6 REF;
- LM head >=1.50x;
- Mamba-in >=1.50x;
- no N=4 case below 0.95x;
- REF drift <=7%.

## Primitive B — MRHS256

Files:

- `E100_MRHS256_PREREGISTRATION.md`
- same V3 adopted-baseline addendum/core;
- `mrhs256_exact_kernels.py`
- `e100_mrhs256_v3.py`
- `verify_e100_mrhs256_v3.py`
- `RUN_E100_MRHS256.ps1`

Geometry: one 256-thread block/output row; physical tid is exactly production virtual tid. A thread holds N RHS accumulators, so N=16 costs 16 accumulators/thread instead of the 128 logical accumulators/lane that an ERVF32 N=16 extension would need.

Frozen N=16 full gates:

- all eight families present: attention Q/O, router, Mamba in/out, shared up/down, LM head;
- exact production/adopted/candidate chain;
- N=4 weighted speedup >=1.50x;
- N=16 weighted >=3.0x;
- N=16 LM head >=3.0x;
- N=16 Mamba in/out >=2.5x each;
- no N=16 case below0.95x;
- REF drift <=7%.

N=16 is architecturally interesting because the measured routed union is ~64 unique experts for 96 route positions, but that overlap is not included in the MRHS component speedup.

## Primitive C — routed PAIRBATCH

Files:

- `E100_PAIRBATCH_PREREGISTRATION.md`
- `up_proj_pair_batch_kernels.py`
- `e100_pairbatch.py`
- `verify_e100_pairbatch.py`
- `RUN_E100_PAIRBATCH.ps1`

N=4, P=24 routed `(sequence, route)` pairs. It flattens four existing six-slot up-proj launches into one P-slot launch while choosing X from the correct sequence. It does **not** claim expert weight-byte sharing.

Frozen maps:

- `unique24`: primary launch-flattening arm;
- `n4_typical22`: realistic ~22 unique expert pattern;
- `repeat6`: slot-alias exactness stress.

Primary gate: unique24 speedup >=1.08x, drift<=7%, typical22 >=0.98x, all maps exact plus direct production-reference spot check.

## CPU-only preflight

Can run while the target GPU is occupied:

```powershell
.\pro_research\RUN_E100_PREFLIGHT.ps1
```

It py-compiles all E100 Python files and runs 500 CPU reduction-tree trials without creating a CUDA context.

## GPU run order

Do not run concurrently with Claude.

```powershell
git fetch origin
git switch pro-e100-batch
git pull --ff-only origin pro-e100-batch
.\pro_research\RUN_E100_PRIMITIVES.ps1 -Mode smoke
```

Smoke is fail-closed. Only when all three primitives are technically/correctness clean:

```powershell
.\pro_research\RUN_E100_PRIMITIVES.ps1 -Mode full
.\pro_research\PUSH_E100_RESULTS.ps1
```

Never multiply the three component speedups.

## Integration architecture only if primitives survive

The first combined runtime is chosen from measured evidence, not in advance.

For N=4, a 100 aggregate tok/s target requires <=40 ms per batch tick. For N=16 it permits <=160 ms, but per-sequence latency must remain explicitly reported.

Candidate fixed-N graph-resident tick:

1. `[N,hidden]` hidden/norm/acc plus independent Mamba/KV state;
2. adopted-winning exact MRHS geometry for common matrices;
3. router MRHS -> `[N,128]`, then one exact batched top-k;
4. flatten P=N*6 pairs;
5. one device cache-assign/fetch over P, deduplicating repeated expert ids;
6. routed up PAIRBATCH; later expert-major MRHS only for repeated experts if separately preregistered and measured;
7. batched down pipeline, accumulating six contributions per sequence in original route order;
8. common MRHS LM head -> N independent argmax results;
9. capture fixed-N tick in one CUDA graph so launch count is approximately constant in N.

Only that integrated full-model A/B can establish E100. Component projections are not evidence of end-to-end throughput.
