# PORT80B-T0Y-R canonical router report

Date: 2026-08-13

## Verdict

`canonical_router_pass`, independently verified 7/7 from stored tensors.

On the 16 real official Qwen3-Coder-Next layer-0 router inputs that exposed CPU/CUDA divergence, a fixed arithmetic tree produced:

- 8,192/8,192 FP32 logits bitwise equal between CPU and CUDA;
- 160/160 ordered top-10 expert IDs equal;
- zero different logit bits and zero different ID values;
- two CUDA executions bitwise equal;
- finite logits and valid unique expert IDs;
- approximately 2,230,784 bytes of active input/output buffers;
- no Q5 bank, model forward or host registration.

The construction explicitly performs BF16-to-FP32 conversion, an FP32 round-to-nearest product and a separate FP32 round-to-nearest addition for hidden indices 0 through 2047 in fixed order. FMA contraction is forbidden. Equal logits are resolved by ascending expert ID.

## Why this matters

T0X-R established a concrete official-router portability counterexample: only 14/16 expert sets and 12/16 ordered expert lists agreed CPU↔CUDA. T0Y-R removes the implementation-dependent reduction tree and obtains exact cross-backend outputs on the same real inputs. This is a valid mechanism result and a candidate compiler/runtime primitive.

## Critical limitation

The canonical expert order matched the official CPU router on only 12/16 rows and the official CUDA router on only 11/16 rows. It therefore defines a third deterministic router result; it does not yet preserve the pinned official model's semantics. No language-model quality, loss, logits, throughput or timing acceptance was tested.

Two pre-output compile failures are retained. The first lacked an explicit CUDA include; the second used the derived reference environment whose CuPy header discovery failed before applying the option. Neither launched a kernel or created a scientific result. The final run used the workspace main `.venv`, which is the already-proven CuPy/NVRTC environment for D5-D10, with identical locked source and inputs.

## Next gate

Before considering integration, test canonical routing on many fresh real hidden states and measure:

1. route disagreement against the pinned official CPU reference;
2. layer-output and downstream quality impact;
3. an optimized deterministic reduction tree rather than the deliberately serial oracle;
4. timing overhead and cross-device reproduction.

## Claim boundary

This is an exact deterministic-router primitive pass for 16 layer-0 inputs on one CPU and one RTX PRO 2000 Blackwell stack. It is not an industrial breakthrough or a full-model result.
