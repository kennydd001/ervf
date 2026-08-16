# E100-PAIRBATCH preregistration — fixed-N routed-pair launch flattening

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: **no target-GPU result exists for this experiment**.

## Purpose and claim boundary

For N active sequences and top_k=6, a MoE layer contains P=N*6 routed `(sequence, expert)` pairs. The previous Python prototypes repeatedly entered per-sequence/per-expert loops. E100-PAIRBATCH tests the structural primitive needed by a fixed-N CUDA graph: flatten all P pairs and execute the expert up-projection in one launch, with each pair reading the correct sequence activation and assigned expert cache slot.

This primitive does **not** reduce the bytes of distinct expert matrices. It is a launch/orchestration primitive, not route-union weight sharing. Any duplicate experts are intentionally irrelevant to the primary test; the same expert may occupy the same physical cache slot for multiple pairs and the arithmetic remains independent.

No E100 claim may be made from this component result alone.

## Exactness contract

The candidate kernel is a mechanical extension of the already verified `UpProjBatchKernels.gemv_nvfp4_ervf_ind_batched` body:

- reference: N launches, each launch handles the six route slots of one sequence and one activation vector `X[s]`;
- candidate: one launch with `blockIdx.y = pair`, `seq = pair / top_k`, reading `X[seq]`;
- slot/id/global-scale selection remains per pair;
- every NVFP4 code load, scale lookup, nibble decode, FMA, virtual-thread assignment and reduction operation inside a pair is unchanged;
- output layout is `[P, rows]` in sequence-major route order, exactly the concatenation of the N reference outputs.

No approximation or cross-pair floating-point reduction is allowed.

## Frozen target shape

Primary N = **4**, top_k = 6, P = 24, using a real Lightning routed-expert up-projection shape (`rows=1856`, `cols=2688`) and real checkpoint bytes from one real MoE layer.

Diagnostic N = 2 and N = 8 may be run, but cannot rescue a failed N=4 primary result post hoc.

Three deterministic pair maps are frozen for full mode:

1. `unique24`: 24 distinct expert ids (no overlap), isolating launch flattening;
2. `n4_typical22`: exactly 22 unique experts across 24 pairs, approximating the measured N=4 mean route-union size (~21.7) without depending on a favorable live route;
3. `repeat6`: all four sequences use the same six expert ids, a stress test that must remain exact even when 24 pairs alias six physical cache slots. It is not the primary performance arm.

The primary performance gate uses `unique24` only. This prevents overlap from being smuggled into a launch-batching claim.

## Cache construction

The runner creates a device cache containing the unique expert records required by the frozen pair map, copied from the real pinned routed bank. `slots[p]` maps each pair to its cache record and `ids[p]` holds the real expert id for the global scale lookup. Candidate and reference consume the exact same cache arrays, slots, ids, globals and X values.

## Correctness gates

All mandatory:

- N=4 all three maps bit-identical candidate vs reference for every one of at least three deterministic X batches in full mode;
- deterministic candidate repeat bit-identical;
- direct reference-vs-production spot check for at least one sequence/map against `FusedNVFP4.gemv_ervf_indirect` or the adopted equivalent, so the reference itself is not merely self-consistent;
- no NaN/Inf;
- `repeat6` remains exact despite slot aliasing.

Any failure => `correctness_failed`; no timing interpretation.

## Timing protocol

CUDA-event device time after warmup. Frozen order for each map: `REF, PAIR, PAIR, REF`.

- REF = N sequential calls of the existing six-slot `UpProjBatchKernels.run_batched`.
- PAIR = one P-slot candidate launch.
- full mode: >=4 rounds per arm and >=10 repeats per round.

Report raw samples, midpoint reference, reference drift and pair speedup.

## Primary N=4 performance gate

An E100-useful structural primitive requires on `unique24`:

- exactness/determinism gates pass;
- `pair_speedup >= 1.08x`;
- reference A/B drift <=7%;
- candidate is not slower on `n4_typical22` (`speedup >= 0.98x`).

The 1.08x threshold is intentionally modest: this primitive eliminates N-1 launches, not matrix bytes. It earns integration only if that structural simplification has a measurable cost benefit before graph capture.

## Interpretation

- `correctness_failed`: stop.
- `pairbatch_null`: exact but primary performance gate fails; keep the kernel only as possible graph-structural code, not as a speed primitive.
- `pairbatch_candidate`: exact and primary gates pass; eligible for fixed-N=4 graph/runtime integration.

PAIRBATCH and MRHS are orthogonal: MRHS reuses common matrix traffic across sequences, while PAIRBATCH flattens different routed expert work. Their speedups must never be multiplied. Only a later combined full-model A/B may establish their joint benefit.
