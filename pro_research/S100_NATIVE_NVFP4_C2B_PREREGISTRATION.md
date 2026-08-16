# S100 native NVFP4 C2b — isolated current-Torch execution gate

Date: 2026-08-16
Branch: `pro-s100-nativefp4-c2b`
Status: preregistered before local execution.

## Why C2b exists

C2 did **not** test native FP4 hardware. It stopped at the Python API gate because the known-working Nemotron environment contains PyTorch `2.9.1+cu128`: `torch.float4_e2m1fn_x2` exists there, but public `torch.nn.functional.scaled_mm`, `ScalingType.BlockWise1x16`, `SwizzleType.SWIZZLE_32_4_4`, and `_scaled_mm_v2` do not.

Current official PyTorch 2.12.1 provides that contract and an official Windows CPython 3.12 CUDA 13.2 wheel. C2b therefore isolates the software-stack change in a **new virtual environment**. It must not modify `.venv-nemotron`.

C2b is still only a synthetic hardware/API/timing experiment. It makes no Lightning quality, token/s, or end-to-end claim.

## Frozen environment change

The runner must use:

- a separate `.venv-fp4-c2b`;
- the repository's current Python 3.12 launcher/interpreter;
- `torch==2.12.1` from the official `cu132` PyTorch wheel index;
- no change to `.venv-nemotron`.

The result records the installed Torch version, Torch CUDA runtime, GPU capability and the `_scaled_mm_v2` schema.

## Frozen synthetic numerical case

Data values are intentionally trivial so scale permutation cannot fake a numerical pass:

- packed E2M1 FP4 A and B bytes: `0x22`, i.e. two exact `+1` values per byte;
- local E4M3 block scales: exact `+1`;
- one local scale per 16 K values (`BlockWise1x16`);
- `SWIZZLE_32_4_4` is declared to the PyTorch operation;
- BF16 output;
- no tensor-wide/global second-level scale in C2b;
- expected result for every output element is exactly BF16(`K`).

The all-one scale values are permutation invariant. C2b is therefore a hardware execution/shape/contract test, **not** proof that arbitrary real checkpoint scales have been swizzled correctly. That belongs to C3.

Physical local-scale storage follows the already frozen C1 convention:

- A scale storage: `[ceil(M,128), ceil(K/16,4)]`;
- B natural storage: `[ceil(N,128), ceil(K/16,4)]`, exposed transposed to the GEMM as `[ceil(K/16,4), ceil(N,128)]`.

No alternative layouts may be tried after seeing timing results. If this documented/frozen contract is rejected, record the rejection and move to a direct CUTLASS contract branch rather than layout fishing inside C2b.

## Gates

### API / provenance

- G1: Torch version begins `2.12.1+cu132` (or `2.12.1` with `torch.version.cuda == 13.2`).
- G2: CUDA available and device capability exactly/at least 12.0.
- G3: `F.scaled_mm` exists.
- G4: FP4 dtype exists.
- G5: `ScalingType.BlockWise1x16` exists.
- G6: `SwizzleType.SWIZZLE_32_4_4` exists.
- G7: `_scaled_mm_v2` exists/schema can be recorded.

If any API/provenance gate fails, stop before performance shapes.

### Known-value correctness

Run `(M,N,K)` = `(1,128,256)`, `(2,128,256)`, `(16,128,256)`, `(128,128,256)`.

For every successfully executed known case:

- output is finite;
- two independent calls are bit-identical (`torch.equal`);
- all outputs equal exact BF16(`K`).

Decisive gate K1: `M=1` must execute. If it does not, no real-shape timing claim is allowed.

### Real-shape synthetic performance

Only after all API + known-value correctness gates pass:

- `M1_QLIKE`: M=1, N=4096, K=2688
- `M2_QLIKE`: M=2, N=4096, K=2688
- `M1_MAMBA_IN`: M=1, N=10304, K=2688
- `M2_MAMBA_IN`: M=2, N=10304, K=2688
- `M1_LM_HEAD`: M=1, N=131072, K=2688
- `M2_LM_HEAD`: M=2, N=131072, K=2688

Each timing is CUDA-event timing after warm-up. No CPU wall-clock token/s projection is a pass gate.

Performance gates retained from C2:

- P1: Q-like M1 < 0.20 ms
- P2: Q-like M2 < 0.25 ms
- P3: Q-like M2/M1 <= 1.40
- P4: Mamba-in M1 < 0.30 ms
- P5: Mamba-in M2/M1 <= 1.40

C3 may open if all correctness gates pass and at least one decisive real-shape family demonstrates both a useful M1 absolute time and M2/M1 <=1.40.

LM-head is diagnostic and not mandatory because its very large N may be memory/capacity sensitive on 8GB.

## Status vocabulary

- `api_contract_failed`: software/API/provenance gate failed; no hardware conclusion.
- `native_execution_failed`: correct API exists but frozen known-value M1 cannot execute.
- `native_executes_below_c3_gate`: correctness passes, performance gate does not.
- `native_execution_candidate`: correctness passes and C3 performance-opening condition passes.
- `technical_failure`: harness failure.

## Claim discipline

A C2b `native_execution_candidate` result means only that a native SM120 FP4 path deserves C3 integration work. It does **not** mean:

- the real Lightning checkpoint is numerically equivalent under W4A4;
- real per-tensor/global NVFP4 scaling is correct;
- activation quantization cost is small;
- the end-to-end model is faster;
- 100 tok/s has been reached.

C3 must separately validate real weights/scales, any tensor-wide second-level scaling, activation quantization, token/quality gates, and integrated physical timing.
