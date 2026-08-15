# HET-NEXT-L0-PV0-R2 — selected real-weight process validation

## Status and claim boundary

This immutable design supersedes PV0-R1 only for future work. Earlier documents
and their NO-GO audits stay immutable. PV0-R2 authorizes no runner, executable
preflight, quantization, compiler or device call. The read-only design audit used
only the official shard header and the immutable S0-R5 evidence JSON to create
the small manifest bound below; no source tensor payload or model was read.

The only admissible positive claim is:

> For the known official Qwen3-Coder-Next layer-0 `p0/n16` route, process-isolated
> Intel and NVIDIA workers reproduced an independent Q5/ERVG oracle for the ten
> selected real experts plus shared, and their official-order token-15 component
> aggregate met the preregistered numerical envelope.

This is known-row validation, not held-out evidence, full-p0 or layer equality,
quality, performance, overlap, acceleration, deployment, novelty or breakthrough.

## Immutable bindings and source manifest

PV0-R2 incorporates the exact evidence bindings and header-only D2 ranges from
PV0-R1 SHA-256
`950ef93b4fe054dd6c1f429bf2856e689cf4544c9ed6829067baae38648d4368`,
except where superseded below. Its implementation design SHA-256 is
`00ae9bcfe8cebb1e5b45260cb7d28be0d3f9cdb7aa5dd90354cbe32f392afa15`.

The normative machine-readable 33-record manifest is
`reports/streamq5_moe/het_next_l0_pv0r2_selected_source_manifest.json`, 22,287
bytes, SHA-256
`0e8882943590e5bb5c9a9d26bdb89e90963c6f732e707bae78f6f50c18cfee40`.
It enumerates every canonical source key, expert, projection, BF16 shape,
relative/absolute half-open offset, exact 2,097,152 source bytes and source,
codes, scales, combined and decoded SHA-256. The 33 records cover exactly the
ten route experts `8,12,50,168,199,237,239,245,374,474` and shared `512`;
source bytes total 69,206,016. Offsets came from the 194,000-byte official shard
header; hashes came from S0-R5 result SHA-256
`56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91`.
Future source and verifier must match the manifest exactly; reconstruction from
a same-shaped or adjacent tensor is forbidden.

The D2 reader remains header-only and range allowlisted. Its captured
`p0_whole_shared_gate` is an explicit package input: BF16 `[16,1]`, absolute
range `[155209092,155209124)`, SHA-256
`3630e2b1cb0ad297f0efd2f029140f5befd810c3520c4dc7eeb0ce746ed49fc0`.
It is the official **linear** outer-gate output. No 34th checkpoint weight is
read. No p1-p3 value range is opened, mapped or hashed.

## Frozen selected graph and source control

Route IDs, native BF16 weights, Intel rank0-3/NVIDIA rank4-9+shared ownership,
the exact 23 selected p0 hit pairs and ascending expert-ID merge order from R1
remain binding. Only the selected subgraph is computed at tokens 0-14; token 15
contains exactly the selected top ten and is the sole scored row.

The graph-matched source control uses the official PyTorch order:

1. `topk_position,token_index=torch.where(mask[expert])` in its observed order;
2. concatenate source BF16 gate then up weights; one fused `F.linear`;
3. split gate then up; BF16 `F.silu(gate,inplace=False) * up`;
4. BF16 down `F.linear`;
5. exact operand order `down_bf16 * route_weight_bf16`; materialize that result
   BF16 before `index_add_` into a BF16 zero buffer;
6. experts loop in ascending ID `8,12,50,168,199,237,239,245,374,474`;
7. shared uses fused gate/up, BF16 SiLU-times-up and BF16 down;
8. outer gate is exactly `torch.sigmoid(captured_shared_gate_linear_bf16)`, which
   must return BF16; gated shared is exact operand order
   `outer_sigmoid_bf16 * shared_raw_bf16`, returning BF16.

For p0, immutable S0-R5 source arrays already equal D2 by SHA:
`p0_source_routed` and D2 experts both
`a74a8a9ef47df5a43ff6ca3ecd28a14650c6275586b21ef6c0fc9f1c3559477c`;
`p0_source_shared_raw` and D2 shared both
`3e1f0052460430ca03c19f7a312a80c68034d86b387d3981ae0cce3224e67125`.
PV0-R2 independently repeats this exact graph and requires full-16 bitwise
equality for routed and shared raw, plus token-15 equality. This is a hard
`source_graph_negative` gate. It does not borrow whole/prefix diagnostic bounds.

## Exact Q5 arithmetic

For each row-major group of 128 source BF16 values, convert to FP32, compute
FP32 `max(abs(group))`, then FP32 scale `maxabs/15`. If maxabs is zero, use FP32
scale 1 and q0. Otherwise calculate `source_fp32/scale_fp32`, call
`torch.round` in the bound CPU runtime—round-to-nearest, ties-to-even—then clamp
integer q to `[-15,15]`. Convert scale to BF16 once for storage. Fields are q+15
in `[0,30]`, field31 forbidden, eight little-order fields/five bytes. Decode is
`float32(q) * float32(bf16_scale)` and one BF16 cast. Code/scales/decoded bytes
must match every manifest digest. Total compressed bytes remain 22,167,552.

The independent CPU Q5/ERVG oracle uses width-8 reductions and BF16 endpoints.
Per expert, gate/up are stored BF16, BF16 SiLU and BF16 multiplication create
activation, down is stored BF16, then exact BF16
`down_bf16 * route_weight_bf16`. The weighted tensor is cast/stored BF16 before
ascending-ID BF16 `index_add_`. Shared outer sigmoid/multiply semantics are
identical to the source control. Every intermediate and accumulator is retained.

## Device and numerical gates

Intel keeps records, input, gate, up, SiLU, activation, down and weighted output
in host-USM; all arguments use `clSetKernelArgMemPointerINTEL`, and the CPU reads
output directly after `clFinish`. All weight/input/output cl-buffer, read, write,
copy and migrate call counts are zero.

NVIDIA receives 21 records—six routed triplets plus three shared projections—
total 14,106,624 bytes. Its exact allowed allocations are:

| object | pointer class | bytes | inbound/outbound copy |
|---|---|---:|---|
| pinned record package | host pinned | 14,106,624 | coordinator RAM -> pinned once |
| device Q5 staging | CUDA device | 14,106,624 | pinned -> device once before compute |
| p0 input | CUDA device BF16 | 65,536 | package input -> device once |
| gate | CUDA device BF16 | 11,264 | none; kernel output |
| up | CUDA device BF16 | 11,264 | none; kernel output |
| SiLU | CUDA device BF16 | 11,264 | none; kernel output |
| activation | CUDA device BF16 | 11,264 | none; kernel output and direct down input |
| routed down | CUDA device BF16 | 45,056 | none; kernel output |
| shared down/raw | CUDA device BF16 | 65,536 | none; kernel output |
| weighted token15 + shared gated | CUDA device BF16 | 28,672 | none; kernel output |
| retained stage copy | pinned host | 221,184 | device -> pinned once after sync |

The 11 routed hits equal 11,264 bytes per `[hits,512]` stage and 45,056 bytes
for `[hits,2048]` down. Shared full16 uses 16,384 bytes for each 512-wide stage
and 65,536 bytes down. The retained-stage copy is routed
gate/up/SiLU/activation/down (90,112 bytes) plus shared
gate/up/SiLU/activation/down (131,072 bytes), total exactly 221,184. The exact
sum of all rows in the table is 28,684,288 bytes. Static preflight must derive
both sums from the machine-readable allocation table and reject disagreement.
No pool, dense dequantized
weight matrix, CPU activation input or unlisted pointer is allowed. Compiler
source, pointer IDs, offsets, copy direction/bytes and allocation/free ledger are
raw evidence.

The captured shared-gate-linear BF16 `[16,1]` is copied with p0 input. NVIDIA
computes its sigmoid in its bound CUDA implementation, stores BF16, then launches
gate-first BF16 multiplication `sigmoid_bf16 * shared_raw_bf16`; CPU source/Q5
oracles compute the exact PyTorch counterpart. Device gate/up must be bitwise
CPU-Q5. Device SiLU, activation, down, weighted, shared sigmoid/raw/gated and
heterogeneous merge must be finite with `relL2<=0.001` versus CPU-Q5. Q5 and
heterogeneous routed/shared raw/shared gated must independently meet unchanged
source-quality `relL2<=0.08`.

## Phase-bounded resources

Processes never form one monolithic peak:

1. Builder phase: coordinator plus CPU builder only; peak combined RSS <=4 GiB.
   It exits completely after writing/fsyncing the 22,167,552-byte packages and
   small oracle bundle. Post-exit survivor count is zero and available RAM
   remains >=2 GiB.
2. Device phase: coordinator plus exactly two children overlap; CPU builder and
   verifier are absent. Each child RSS <=1.5 GiB; coordinator <=512 MiB;
   aggregate sampled process-tree RSS <=3.5 GiB. Intel allocations <=16 MiB;
   NVIDIA pinned+device allocations <=64 MiB; available RAM >=2 GiB.
3. Verifier phase: both children have exited and all device resources are freed;
   verifier plus coordinator only, aggregate RSS <=4 GiB and available RAM
   >=2 GiB. Verification streams one projection and retains no decoded bank.

Peak working set and current RSS are sampled at start, after every projection,
post-package, child-ready, each child-result, post-child-cleanup, each verifier
projection and final cleanup. A phase overlap or cap violation is
`blocked_resource`; there is no retry. Final evidence remains <=8 MiB.

## Process, controls and adjudication

R1's standalone suspended-child/Job-Object, canonical framed-message,
QPC-timeout, no-survivor, independent-verifier-before-commit and atomic failure
protocol remains exact. CAP0X is evidence, not harness inheritance.

Safe controls remain wrong expert, projection swap, wrong owner/rank, changed
digest, field31 and sealed D2 range. Before children launch, each backend's
lexicographic down-record/row/column one-step-toward-zero and first k=-8..8
one-hot witness is fully frozen with coordinate, q/q-prime, scale, code digests,
activation SHA and original/mutated output words. Checker rejection precedes one
unsafe device reproduction. Stored booleans are never trusted.

Outcomes remain `real_weight_process_validation_positive`,
`source_graph_negative`, `codec_negative`, `device_numerical_negative`,
`quality_negative`, `control_negative`, `invalid_protocol`,
`blocked_capability`, or `blocked_resource`. One clean future attempt only,
after implementation audit, static preflight and separate execution GO.
