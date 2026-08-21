# S100 Phase66: Ornith 65 tok/s measured-component budget

This adjudication consumes only already-written Phase49/58/59/62/64/65 JSON.
No constants are remeasured or selected after seeing this calculation.

The known hot floor is:

- all 30 linear and 10 full FP8 attention projections from Phase58;
- 40 worst-case-unique routed bulk dispatches from Phase59;
- 40 shared M4 experts from Phase49 (Phase65 overlap is excluded unless green);
- the Phase64 native top-64 plus exact ERVF rerank head.

The 65 tok/s boundary is 61.53846 ms/H4. The residual is explicitly reserved
for unmeasured router, linear/full-attention cores, norms, residuals, route
weighting/reduction, final norm, argmax and orchestration. A positive residual
is necessary but not sufficient for 65 tok/s.

Phase62's exact miss-count points are combined with an interpolated hot-hit
curve to report uniform misses per layer. No end-to-end claim is allowed.
