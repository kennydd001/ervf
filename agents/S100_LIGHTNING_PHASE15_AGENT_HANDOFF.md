# S100 Lightning Phase 15 — agent handoff

## Material correction

Nano and Lightning have identical checked architecture shapes, but not identical
weights, routes, states or logits. The inherited Nano quality trace cannot
adjudicate a Lightning candidate.

The first Phase-15 control runs the unmodified Lightning parent against the
inherited trace. A mismatch quarantines the inherited trace. Phase 15 then
creates a new Lightning-parent trace with checkpoint identity embedded.

## Central hypothesis

The Phase-14R native path rounded:

- FP32 activation -> BF16;
- Tensor-Core result -> BF16;
- BF16 result -> FP32 container.

That is not the current ERVF numerical model.

Phase 15 uses FP32 Tensor-Core output and tests a split activation:

    x_hi = BF16(x)
    x_lo = BF16(x - float(x_hi))
    y = GEMM([x_hi, x_lo], W^T, out_dtype=FP32)
    output = y_hi + y_lo

One weight matrix serves both activation terms in one multi-row GEMM.

## Frozen family candidates

- current ERVF with BF16-rounded input: input-rounding attribution;
- TC1 KVO;
- TC2 KVO;
- TC3 KVO;
- TC1 K / V / O;
- TC2 K / V / O / KV / KO / VO.

Q remains on the current QFAST-NVFP4 path.

## Downstream reset

Nano-derived block-verifier, route-union and DFlash2 acceptance results are
quarantined for Lightning. A Lightning block-verifier rerun opens only after a
fresh parent baseline and at least one quality-green native candidate.
