# S100 Phase 14 — survivor adjudication on the same checkpoint

Date: 2026-08-18
Parent: `agent/s100-phase13-subspace-entropy`
Target constraint: `models/nemotron_3_5_lightning`; no replacement checkpoint.

## Status inherited from Phase 13

- 13A lossless entropy is closed for the tested encodings: no useful byte reduction.
- 13C temporal delta is closed for the tested 99%-energy gate.
- 13D native BF16 Tensor-Core block execution is the strongest survivor: the B=4 component ceiling is 5.47x, but no model-level quality validation exists.
- 13B is not a promotion and not a no-go. It measured pooled input PCA only and explicitly omitted WU output error, token fidelity and GPU timing.
- 13E's naive raw-code expert SVD is a negative implementation, not a theorem. Rank-32 reconstruction NRMSE near 0.40 is too poor; only one decoded, activation-weighted sparse-residual kill test is justified.

## Track D — decisive native-BF16 model-quality gate

1. Extend the component benchmark from the six BF16 Mamba blocks to every live BF16 projection.
2. Replace all current BF16 GEMVs in an eager copy of the quality-green parent with native BF16 CUDA/Torch matmul using the same checkpoint weights.
3. Run frozen validation first and heldout only after strict validation passes.
4. Record official fidelity, deterministic repeat, finite outputs and per-domain metrics.

`NATIVE_BLOCK_RUNTIME_BUILD_OPEN=true` only if the extended B=4 component remains >=2.5x and heldout official quality passes. This does not yet claim B=4 end-to-end speed; it authorizes the layer-major block runtime.

## Track B2 — output-aware reduced-rank regression

Repeat activation-subspace research per representative layer and projection, but optimize actual layer output instead of input reconstruction. For captured pairs X->Y, fit reduced-rank regression factors T,C such that Y≈XTC. Frozen ranks are 32/64/128/192/256/384. Report validation output NRMSE, cosine, p95 row error, argmax agreement and physical factor bytes.

`SUBSPACE_RUNTIME_BUILD_OPEN=true` only if a family removes >=35% physical bytes and at least 80% of representative matrices meet NRMSE<=0.03, cosine>=0.9995 and p95 row-relative error<=0.08.

## Track E2 — decoded activation-weighted expert basis

Decode the actual NVFP4 expert values for early/middle/late MoE layers. Weight coordinates using real calibration/validation hidden-state energy. Fit expert-axis ranks 4/8/16/32 and frozen sparse residual fractions 6.25/12.5/25%. Evaluate validation weighted weight error and sampled GEMV output error.

`EXPERT_BASIS_RUNTIME_BUILD_OPEN=true` only if byte ratio<=0.70 and validation output NRMSE<=0.05 with cosine>=0.999. Otherwise expert shared-basis work closes.

## Decision

Phase 14 does not rerun 13A or 13C. Track D receives full validation effort. Tracks B2 and E2 are bounded screens. Missing evidence is reported as incomplete, never silently converted to a no-go.
