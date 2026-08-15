# HET-NEXT-L0-PV0-R3 — selected real-weight process validation

## Status and inheritance

This immutable design supersedes PV0-R2 only for future work. R2 remains an
immutable design NO-GO. R3 authorizes no runner, executable preflight, payload
read, quantization, compiler or device call.

Except for the two exact corrections below, R3 incorporates:

- R2 preregistration, 10,146 bytes, SHA-256
  `66c243b2b0ec52ff1f9cea385c669a0f7256f7bc76e0fee1d829a1d0c45c0fe6`;
- R2 implementation design, 3,947 bytes, SHA-256
  `5fa36c2b19c161f86386ad0f2fa9e815bb9376a18b4d74da2f34986392033cca`;
- the unchanged current 33-record source manifest, 22,287 bytes, SHA-256
  `0e8882943590e5bb5c9a9d26bdb89e90963c6f732e707bae78f6f50c18cfee40`.

The only possible positive claim remains known-row, real-weight,
process-isolated heterogeneous **component validation** for p0/n16. It is not
held-out, full-p0, layer/logit, quality, performance, overlap, acceleration,
deployment, novelty or breakthrough evidence.

## Correction 1 — exact source-reference boundary

The full-16 selected-expert routed subgraph contains exactly the frozen 23 hits
from the ten experts selected at token 15. It does not contain the other p0
experts needed to reproduce the official full-16 routed aggregate. Therefore:

1. The builder creates the selected-source full-16 subgraph with the frozen
   fused `[gate;up]`, `torch.where`, BF16 SiLU/multiply/down, route-weight
   multiply and ascending-expert-ID `index_add_` semantics.
2. The independent verifier separately rebuilds the same selected-source graph
   from the official shard and header-only D2 input/routes without importing any
   builder helper. Builder and verifier must be bitwise equal for every retained
   selected-expert gate/up/SiLU/activation/down, weighted contribution and
   selected routed accumulator state across all 23 hits. This is the full-16
   selected-subgraph self-control.
3. **Only token 15 routed** is compared to official D2
   `p0_whole_experts[15]`. Because its top ten exactly equal the selected ten,
   the builder and independent selected-source token-15 aggregate must both be
   bitwise equal to the D2 4,096-byte BF16 row. Tokens 0-14 are never compared
   with the full D2 routed aggregate.
4. Shared is complete at every token. Builder and independent source replay
   `shared_raw[0:16]` must both be bitwise equal to the entire official D2
   `p0_whole_shared` BF16 `[16,2048]` array, SHA-256
   `3e1f0052460430ca03c19f7a312a80c68034d86b387d3981ae0cce3224e67125`.
   Shared-gated is independently derived from captured gate-linear plus shared
   raw; D2 does not supply it as a separate official oracle.

Any mismatch is `source_graph_negative`. No whole/prefix tolerance is used and
no missing selected subgraph is silently treated as the full p0 MoE.

## Correction 2 — exact NVIDIA shared-gate and allocation table

The allowed D2 captured input `p0_whole_shared_gate` is the 32-byte BF16
`[16,1]` **linear** outer-gate array at absolute range
`[155209092,155209124)`, SHA-256
`3630e2b1cb0ad297f0efd2f029140f5befd810c3520c4dc7eeb0ce746ed49fc0`.
It travels as a separate 32-byte pinned input and a separate 32-byte CUDA-device
linear-gate allocation. NVIDIA computes `sigmoid(linear_bf16)` on device into a
distinct 32-byte BF16 allocation, then computes exact operand order
`sigmoid_bf16 * shared_raw_bf16`. It retains both 16-word arrays. No checkpoint
outer-gate weight is read.

The complete normative NVIDIA allocation table is:

| object | class | bytes | only legal transfer/use |
|---|---|---:|---|
| record package | pinned host | 14,106,624 | coordinator -> pinned once |
| Q5 staging | CUDA device | 14,106,624 | pinned -> device once |
| p0 post-norm input | CUDA device BF16 | 65,536 | package -> device once |
| captured shared-gate linear input | pinned host BF16 | 32 | coordinator -> pinned once |
| captured shared-gate linear | CUDA device BF16 | 32 | pinned -> device once |
| routed gate | CUDA device BF16 | 11,264 | device kernel output |
| routed up | CUDA device BF16 | 11,264 | device kernel output |
| routed SiLU | CUDA device BF16 | 11,264 | device kernel output |
| routed activation | CUDA device BF16 | 11,264 | device output/direct down input |
| routed down | CUDA device BF16 | 45,056 | device kernel output |
| shared down/raw | CUDA device BF16 | 65,536 | device kernel output |
| shared sigmoid | CUDA device BF16 | 32 | device sigmoid output |
| six weighted routed + shared gated | CUDA device BF16 | 28,672 | device output |
| all retained routed/shared stages | pinned host | 221,184 | device -> pinned once after sync |
| weighted/shared-gated + sigmoid outbound | pinned host | 28,704 | device -> pinned once after sync |

The exact sum is **28,713,088 bytes**. The 221,184-byte retained stage row is
routed gate/up/SiLU/activation/down `90,112` plus shared
gate/up/SiLU/activation/down `131,072`. The 28,704-byte outbound row is six
weighted routed vectors `24,576` plus shared gated `4,096` plus shared sigmoid
`32`. The source implementation must encode this table as data; static preflight
and independent verifier recompute every row and both sums. No pool, alias,
unlisted allocation, dense dequantized weight, CPU activation substitution or
unlisted copy is legal.

## Unchanged gates and future authorization

All R2 Q5 ties-to-even/zero/packing semantics, exact route multiply/cast and
ascending merge, Intel all-host-USM/no-copy contract, device gate/up bitwise
gate, downstream `relL2<=0.001`, source-quality `relL2<=0.08`, controls,
header-only p0 seal, standalone Job/framed/timeout/no-survivor protocol, phased
RSS/resource gates, independent-verifier-before-commit and one-attempt rule
remain unchanged.

Outcomes remain `real_weight_process_validation_positive`,
`source_graph_negative`, `codec_negative`, `device_numerical_negative`,
`quality_negative`, `control_negative`, `invalid_protocol`,
`blocked_capability`, or `blocked_resource`. Implementation begins only after a
new independent design GO; physical execution needs later source audit,
preflight and explicit GO.
