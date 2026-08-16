# S100 native NVFP4 C2 — PyTorch SM120 execution-contract preregistration

Date: 2026-08-16
Branch: `pro-s100-nativefp4-c2`
Base: `pro-s100-nativefp4@8606420844132f198d49157ea792150d5ab19b0e`
Frozen before C2 execution.

## Established inputs

C0/C0B/C1 already established on the exact local Lightning checkpoint:

- SM120 Blackwell hardware and CUDA/cuBLASLt capability are present;
- 5,935 NVFP4 weight/scale pairs are logically group-16 E2M1 + E4M3;
- the checkpoint scale layout can be losslessly permuted to `SWIZZLE_32_4_4`;
- C1 inverse roundtrip, code bytes, global scales and sampled dequant reconstruction are exact;
- total native scale-layout padding is 1.6778%.

C2 does **not** re-audit those facts. It asks the next narrow question: can the locally installed PyTorch eager scaled-matmul path actually execute native FP4 block-scaled GEMM on this SM120 laptop, including the batch-1 shape class relevant to decode?

## Why a synthetic known-value probe first

A real Lightning native-FP4 target path also needs activation NVFP4 quantization, a global scale convention and a numerical/quality gate. Those choices must not be mixed into the hardware execution question.

C2 therefore uses exact synthetic E2M1 data representing `+1`, exact E4M3 block scales representing `1`, and no non-unit global scale. If the native operation is correctly wired, an all-ones `[M,K] @ [K,N]` product must produce `K` in every output element (subject only to the requested BF16 output representation). This isolates dtype/layout/API/kernel support.

## Frozen API

Preferred API:

```python
torch.nn.functional.scaled_mm(
    A_fp4,
    B_fp4,
    scale_a=A_scale,
    scale_recipe_a=ScalingType.BlockWise1x16,
    scale_b=B_scale,
    scale_recipe_b=ScalingType.BlockWise1x16,
    swizzle_a=SwizzleType.SWIZZLE_32_4_4,
    swizzle_b=SwizzleType.SWIZZLE_32_4_4,
    output_dtype=torch.bfloat16,
)
```

No `torch.compile`: current PyTorch routes swizzled block-scaled recipes through eager/native handling, and C2 is testing that contract directly.

Packed data:

- E2M1 `+1` code = nibble `0x2`;
- two values/byte therefore use `0x22`;
- packed uint8 storage is viewed as `torch.float4_e2m1fn_x2`;
- `A` is row-major;
- `B` is created as natural `[N,K]` packed rows and transposed to logical `[K,N]`, preserving the column-major RHS convention expected by native scaled GEMM.

Block scales:

- local scale is exact E4M3 `1.0`;
- natural group is 16 K elements;
- native C1 layout is `SWIZZLE_32_4_4`;
- for the all-one scale test the byte permutation is value-invariant, but C2 still allocates the C1 padded physical shape/count;
- A scale is `[ceil(M/128)*128, K/16]` when `K/16` is a multiple of 4; if not, K-scale count is padded to the next multiple of 4;
- B natural scale rows are N; after native row padding the scale tensor is transposed for the logical `[K,N]` RHS.

There is one frozen contract only. If PyTorch rejects this exact shape/layout contract, C2 reports the exact exception and status `api_contract_failed`; it does not silently try alternate layouts in the same experiment.

## Test shapes

Known-value correctness:

- `(M,N,K) = (1,128,256)` — decisive batch-1 support;
- `(2,128,256)`;
- `(16,128,256)`;
- `(128,128,256)`.

Decode-relevant performance shapes (all-one data, no activation quantization cost):

- `M1_QLIKE = (1,4096,2688)`;
- `M2_QLIKE = (2,4096,2688)`;
- `M1_MAMBA_IN = (1,10304,2688)`;
- `M2_MAMBA_IN = (2,10304,2688)`;
- `M1_LM_HEAD = (1,131072,2688)` if memory allocation fits; otherwise record `not_run_memory_gate`, not failure;
- `M2_LM_HEAD = (2,131072,2688)` under the same memory gate.

These performance shapes use synthetic already-quantized FP4 B operands. They measure native GEMM only, **not checkpoint loading/repack and not activation quantization**.

## Correctness gates

- `G1_public_scaled_mm_present`;
- `G2_fp4_dtype_present`;
- `G3_scaling_and_swizzle_enums_present`;
- `G4_M1_known_value_executes`;
- `G5_all_executed_known_value_outputs_equal_expected_bf16`;
- `G6_deterministic_repeat`;
- `G7_no_nan_inf`.

If G1-G4 fail, native PyTorch execution is not usable through this contract. This does **not** negate C1 or prove native hardware impossible; a CUTLASS/vLLM/FlashInfer path remains distinct.

## Performance reporting / gates

CUDA-event timing after compile/warmup, 100 repetitions per small/QLIKE shape and 30 for LM head.

Report p50-equivalent average event ms and effective physical matrix payload GB/s.

Exploratory opening gates:

- `P1_M1_QLIKE_lt_0_20ms`;
- `P2_M2_QLIKE_lt_0_25ms`;
- `P3_M2_vs_M1_QLIKE_time_ratio_le_1_40` — direct evidence that two RHS can amortize the same FP4 weight stream;
- `P4_M1_MAMBA_IN_lt_0_30ms`;
- `P5_M2_vs_M1_MAMBA_IN_ratio_le_1_40`.

These are hardware-path gates only. No full-model integration is authorized by speed alone.

## Next-phase rule

C3 real-checkpoint/native activation work opens only if all correctness gates pass and at least P1+P3 or P4+P5 pass.

C3 must separately preregister:

- activation NVFP4 quantization recipe;
- global-scale/alpha semantics;
- comparison to current V18 FP32/BF16 activations;
- layer/logit/token quality gates;
- activation-quantization runtime cost.

No 100 tok/s claim is permitted from C2.
