# S100 Phase 13F — Subspace-Residual ERVF

This test measures the core equation on real calibration and validation
activations from the same Nemotron checkpoint:

`y ≈ (W U)(Uᵀ x)`

For validation tokens that exceed a calibration residual threshold, the test
models an exact `W x` fallback. The test covers all resident BF16 Mamba
input/output matrices. It measures residual energy, actual output NRMSE, and
the fast-token fraction for several fallback targets.

This is still a component screen. It does not alter the runtime, preserve
Mamba state under approximate updates, test refresh intervals, or run official
heldout generation quality. Therefore the promotion flag remains closed.

At rank 256, the validation output NRMSE without fallback averaged about
0.375 for Mamba input projections and 0.587 for Mamba output projections.
Even with a calibration threshold intended to retain 50% of tokens on the
fast path, validation fast fractions averaged only about 2.4% and 2.3%
respectively. This falsifies the simple global low-rank fast path for these
Mamba matrices; a useful system would need a much more selective gate or a
different basis/residual representation.
