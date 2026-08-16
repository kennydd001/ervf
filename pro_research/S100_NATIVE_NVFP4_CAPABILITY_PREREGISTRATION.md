# S100 native Blackwell NVFP4 capability audit — preregistration

Date: 2026-08-16
Branch: `pro-s100-nativefp4`
Base: `pro-research@43720efbb202c115b49413e13157dad4867093bf` (V19; V18 remains the adopted record)
Status at freeze: **no native Tensor Core NVFP4 result exists in this project**.

## Motivation

The current Lightning runtime interprets packed E2M1 weights and per-16 E4M3 block scales in custom CUDA-core GEMV kernels. The target GPU is Blackwell-generation and the installed CUDA family is 12.9. NVIDIA cuBLASLt/CUTLASS support Blackwell block-scaled FP4 Tensor Core GEMM using E2M1 values with 16-element UE4M3 block scales.

This does **not** imply the existing checkpoint can be dropped into cuBLASLt unchanged. Native block-scaled FP4 GEMM also constrains operand types/layouts, and the official SM120 CUTLASS NVFP4 example uses NVFP4 for both A and B. Quantizing the runtime's FP32 activations to FP4 would change arithmetic and therefore requires a separate numerical/token-equivalence gate.

## Phase-C0 claim boundary

C0 is capability/format inspection only. It may establish:

- exact GPU compute capability and CUDA runtime/driver versions;
- whether `cublasLt64_12.dll` is discoverable/loadable and core cuBLASLt symbols exist;
- installed Torch/nvmath/CuPy versions and whether they expose FP4 helpers/dtypes;
- checkpoint quantization metadata for weight and input-activation group size/type;
- representative raw NVFP4 scale-byte statistics, especially whether bit 7 is ever set;
- representative packed-code and scale tensor shapes/byte counts;
- whether the checkpoint's per-block scale representation is a plausible **value-format** match to UE4M3 group-16.

C0 may NOT claim:

- native matmul compatibility;
- correct scale-factor **layout** for cuBLASLt/CUTLASS;
- numerical equivalence;
- token parity;
- speedup.

## Frozen capability gates

- `C0_GPU_SM120_OR_NEWER`: compute capability major >= 12 for this RTX Blackwell path.
- `C0_CUDA_RUNTIME_GE_12080`: runtime >= 12.8 because Blackwell FP4 cuBLASLt support begins there.
- `C0_CUBLASLT_LOADABLE`: Windows cuBLASLt DLL is found and loads.
- `C0_CUBLASLT_CORE_SYMBOLS`: `cublasLtCreate`, `cublasLtDestroy`, `cublasLtMatmul`, `cublasLtMatmulDescCreate`, `cublasLtMatrixLayoutCreate` exist.
- `C0_MODEL_GROUP16_NVFP4`: local quantization metadata declares 4-bit float weights with group size 16 for the relevant linear group, or the equivalent ModelOpt metadata.
- `C0_SCALE_SIGNBIT_CLEAR_REPRESENTATIVE`: every sampled NVFP4 block-scale byte has bit 7 clear. This is necessary for direct interpretation as unsigned UE4M3 but not sufficient for layout compatibility.
- `C0_REPRESENTATIVE_TENSORS_FOUND`: lm_head/shared/routed NVFP4 scale tensors are found where present in this Lightning checkpoint.

A missing optional Python high-level API (`torch.float4_*`, nvmath) is diagnostic only; direct cuBLASLt remains possible if the DLL/API is present.

## Stop / next step

If any mandatory hardware/library/value-format gate fails, do not build a native FP4 performance arm under this preregistration.

If all mandatory gates pass, phase C1 must be preregistered separately. C1 will:

1. repack only one real Lightning matrix and its block scales to NVIDIA's required scale-factor layout;
2. dynamically quantize one real activation vector/block to NVFP4 using a frozen conversion rule;
3. compare native Tensor Core output against the current V18 arithmetic and quantify numerical error/token risk;
4. time N=1 and N=2 separately;
5. use a control matrix and independent output verification.

No production runtime modification before C1 passes.
