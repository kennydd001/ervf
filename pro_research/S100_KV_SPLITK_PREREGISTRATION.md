# S100 K/V paired split-warp preregistration

Date: 2026-08-16  
Branch: `pro-s100-splitk`  
Base: `pro-research@43720efbb202c115b49413e13157dad4867093bf`

Status at freeze: **no target result exists for this experiment**.

## Question

The six attention layers execute K and V projections with real shape `(256,2688)`.
The adopted selective policy leaves that shape on production BF16 GEMV. In the
captured graph the combined K/V stage was attributed far below the large Q
projection in weight-byte efficiency.

An earlier diagnostic showed that simply putting `(256,2688)` on ERVF-16 is
bit-exact but slower in isolation. That does **not** test the schedule below.

This experiment asks whether retaining every one of production's 256 virtual
threads while splitting its eight reference warps into more schedulable blocks,
and removing the full-X shared-memory staging/barrier, improves the real V18
captured path.

This is a final-mile S100 component. It is not a 100 tok/s claim.

## Frozen candidate

For each attention layer, K and V are paired.

`kv_pair_partial`:

- grid `(row=256, reference-warp-group=2, matrix=K/V)`;
- block = 4 original warps = 128 threads;
- physical lane maps to the exact production virtual `tid`;
- each tid executes exactly `k = tid, tid+256, ...`;
- BF16 widen and `fmaf` order are unchanged;
- each reference warp reduces with exact offsets `16,8,4,2,1`;
- lane 0 writes one FP32 warp sum.

`kv_pair_finalize`:

- one 32-thread warp per `(row,K/V)`;
- lanes 0..7 read the eight reference warp sums, lanes 8..31 read `+0.0`;
- reduction offsets are exactly `16,8,4,2,1`;
- lane 0 writes the final K or V element.

Therefore the floating operation DAG for each output is intended to be identical
to production. Reading X from global/L1/L2 instead of a CTA-local staged copy is
not a numerical change.

K and V together still use two kernel launches: one partial and one finalize.
The production pair uses one complete GEMV launch for K and one for V.

No alternate warp grouping may replace the frozen 4-warps-per-block candidate
after seeing target data.

## Micro correctness/diagnostic

Use all six real attention layers from `nemotron_3_5_lightning_v35`. For each
layer use its real K and V BF16 weights and finite FP32 activations.

Mandatory:

- production K == candidate K bit-for-bit;
- production V == candidate V bit-for-bit;
- candidate repeat deterministic;
- all outputs finite.

The micro timing is diagnostic only. A slow isolated micro result does not stop
the integrated graph test because the hypothesis concerns scheduling inside the
full captured runtime.

## Decisive integrated A/B/A

Baseline is the current record stack, V18 = V6 + H-SCALE + B3.

Order in one runtime/process:

1. `V18_BASE_A`
2. `V18_PLUS_KV_SPLIT`
3. `V18_BASE_B`

Every arm is separately recaptured after installing/restoring only the K/V
dispatch. Same prompt set, exact-state reset and SYNC token semantics as V18.

Smoke: short technical/correctness run.  
Full: 3 prompts, at least 256 generated tokens per prompt.

## Gates

All mandatory:

- G1: candidate token IDs equal BASE_A for every prompt/token.
- G2: BASE_B token IDs equal BASE_A.
- G3: all micro K and V outputs bit-identical and deterministic.
- G4: `abs(BASE_A.p50 - BASE_B.p50) <= 1.0 ms`.
- G5: candidate p50 is at least **0.20 ms/token** below A/B midpoint.
- G6: candidate p50 `< 20.0 ms/token` is reported separately; it is not required
  for G5 because V18 already crosses E50 and the point here is incremental gain.

Interpretation:

- any exactness failure: `correctness_failed`;
- drift failure: `measurement_unstable`;
- exact/stable but <0.20 ms gain: `gate_failed`;
- exact/stable and >=0.20 ms gain: `adoption_candidate`.

No component result is added arithmetically to another projected speedup.
