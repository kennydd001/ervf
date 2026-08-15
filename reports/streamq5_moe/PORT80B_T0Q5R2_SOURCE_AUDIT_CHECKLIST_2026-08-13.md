# T0Q5-R2 independent source-audit checklist

No experiment execution is authorized. Audit the immutable R2 preregistration and codec source before any runner/verifier implementation.

1. Recompute `struct.calcsize(<4sHHHBBIIH2xIII28s)==64`, offsets, CRC chaining, exact zero-reserved bytes/padding and every matrix/expert/bank byte total.
2. Audit RTN ties-to-even, zero-group BF16 scale 1.0, biased q+15, 8-in-5 little order, field-31 rejection and little-endian BF16 scale bytes.
3. Confirm all 1,539 source matrices exist in pinned shard 1 and manifest fields can bind source/header/codes/scales/padding/record/top-level bank.
4. Confirm primary manual arms are full-length 16, graph matched (fused gate_up, captured routes, identical dispatch/reduction/casts), and source-BF16 control gates Q5 interpretation.
5. Confirm shared raw/gated gates and claim boundary: all 512 routed experts get codec/source evidence; numerical quality covers selected natural routes plus shared only.
6. Confirm all 4×4 old/new prompt pairs and all new/new pairs require unequal text/token sequence/token digest, and exactly 16 IDs per row.
7. Confirm each negative control has deterministic row, mutation, safe rejection, unsafe effect requirement and no outcome-driven alternate.
8. Judge frozen thresholds before outputs. Flag any unsupported bitwise/one-ULP graph-control assumption or ambiguity in reconstructed layer formula.

Return GO only for runner/verifier implementation, not for model load, forward or bank build.
