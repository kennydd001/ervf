# S100 phase 6 preregistration — direct downflow + confirmatory sparsity

Date frozen: 2026-08-17

## Parent evidence

- Exact V18: approximately 19.4 ms/token.
- QFAST: 18.75165 ms/token, 53.3286 tok/s, full 10,240-target fidelity green.
- Phase 5 read no heldout candidate.

## Part A — exact CUDA backends

Backends:

1. `legacy`: phase-5 alpha=0, K6 path.
2. `ballot_fused`: exact warp-ballot panel scan plus fused partial reduction and
   route-weight accumulation; existing H-SCALE+B3 gather remains.
3. `direct`: original exact scan/reduction, but the down GEMV reads mapped host
   code bytes directly and does not allocate/copy the VRAM mirror.
4. `direct_opt`: direct host-code GEMV plus ballot scan plus fused reduction.

### Correctness

- all registered smoke prompt token ids equal legacy;
- deterministic candidate repeat;
- finite logits;
- destructive `bad_pick=1` control diverges;
- backend record states K6 and alpha=0.

### Fresh timing

Every backend comparison uses independent processes:

`LEGACY_A -> CAND_A -> CAND_B -> LEGACY_B`.

Full mode requires >=765 samples per arm, <=1 ms baseline/candidate drift and
<=7.8 GiB VRAM. A backend must improve at least 0.15 ms/token to be selected;
otherwise phase 6 retains legacy.

## Part B — fixed validation grid

The following arms are fixed before phase-6 validation:

- `thr_0003`, `thr_0010`, `thr_0015`, `thr_0020`, `thr_0025`;
- `k1`, `k2`, `k3`, `k4`;
- `thr0010_k1`, `thr0010_k2`, `thr0015_k1`.

K maps are fixed from phase-5 calibration order and cannot change after phase-6
validation.

Every arm passing all original official phase-3 validation gates is written to
`S100_PHASE6_CANDIDATES.json`. This is a new confirmatory experiment. Phase-5's
stricter exploratory selection remains recorded and is not relabelled.

## Heldout

All validation-green candidates are evaluated on the untouched `_03/_04`
5,120-target split using the original official gates:

- top1 >=0.95;
- target in top5 >=0.995;
- mean CE delta <=0.05;
- p95 CE delta <=0.25;
- mean coarse KL <=0.02;
- p95 coarse KL <=0.08;
- every domain top1 >=0.90;
- every domain mean CE delta <=0.10;
- finite and deterministic.

## Candidate timing

Only heldout-green candidates are timed. The candidate uses the fastest exact
backend selected in Part A; the two baseline arms remain legacy QFAST. Thus the
measured candidate time includes both exact backend changes and approximate
sparsity.

## Claim boundary

A threshold/K candidate is a compiled QFAST derivative. Exact backend arms do
not change arithmetic. No arm is an S100 result until its complete measured
latency is <=10.000 ms and the frozen heldout gate is green.
