# S100 Phase 14 preregistration

## Frozen order

1. 14D extended native-BF16 ceiling.
2. 14D validation; heldout only if strict validation passes.
3. 14B2 output-aware representative-layer screen.
4. 14E2 decoded expert-basis kill test.
5. Final evidence-completeness-aware summary.

## Data

- calibration `_01`;
- validation `_02`;
- heldout `_03/_04`;
- same Nemotron 3 Nano checkpoint and Phase-9/10 quality parent.

## 14D gates

- real-weight rotation >4x L2;
- B=4 component useful-row speedup >=2.5;
- official validation plus strict validation gates;
- heldout official gates and deterministic repeat;
- finite outputs.

The eager M=1 native runtime is a numerical-fidelity test, not a speed claim. A pass opens construction of the true layer-major B=4 runtime.

## 14B2 gates

Representative early/middle/late Mamba layers plus all attention layers. Reduced-rank regression is fitted only on calibration pairs and evaluated on validation pairs. Per-case pass requires >=35% factor-byte reduction, output NRMSE<=0.03, mean cosine>=0.9995 and p95 relative row error<=0.08. A family opens only when >=80% of its representative cases pass.

## 14E2 gates

Actual decoded NVFP4 up-expert values, ranks 4/8/16/32 and sparse residual fractions 0.0625/0.125/0.25. Calibration activation energy selects the basis/residual mask; validation activation energy evaluates it. Candidate opens only at byte ratio<=0.70, sampled output NRMSE<=0.05 and cosine>=0.999 across early/middle/late layers.

## Claim boundary

No Phase-14 component can claim S100. The final flags only authorize the next runtime build:

- `NATIVE_BLOCK_RUNTIME_BUILD_OPEN`;
- `SUBSPACE_RUNTIME_BUILD_OPEN`;
- `EXPERT_BASIS_RUNTIME_BUILD_OPEN`.
