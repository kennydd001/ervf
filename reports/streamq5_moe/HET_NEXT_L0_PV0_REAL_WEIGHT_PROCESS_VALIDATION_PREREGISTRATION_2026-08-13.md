# HET-NEXT-L0-PV0 — real-weight process-isolated component validation

## Status and single admissible claim

This is a design-only preregistration. It authorizes no runner, executable
preflight, checkpoint scan, quantization, compiler invocation, device
enumeration, allocation, kernel launch or output creation. A separate source
audit must return GO before implementation begins, and a later immutable
execution lock must separately authorize any physical action.

PV0 asks one deliberately narrow question:

> For the already known official Qwen3-Coder-Next layer-0 `p0/n16` route, can
> two process-isolated device workers compute fixed disjoint real Q5 expert
> subsets whose retained intermediates agree with an independently implemented
> CPU Q5/ERGV oracle and whose official-order host merge remains within the
> preregistered numerical envelope of the official BF16 component capture?

A positive result is only a **known-row, real-weight, natural-route,
heterogeneous component validation**. It is not held-out evidence, a layer or
logit reproduction, a quality result, a performance result, simultaneous-kernel
proof, full-model acceleration, deployability, novelty or an industrial
breakthrough.

## Immutable evidence base

Every future PV0 source and lock must bind exact path, byte count and SHA-256.
The normative current bytes are:

- D2-R3 raw/result: `171,696,126`,
  `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`;
  `1,043,105`,
  `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`.
- D2-R3 independent artifact audit/interpretation:
  `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`;
  `be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0`.
- S0-R5 raw/result/commit:
  `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`,
  `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91`,
  `d784ded5e7893095e2f27b75695e635c9cc880109736c87496138e3188509372`.
- C1-R2A raw/result/commit:
  `d7272ce6aec3533b487829360e40398f1d5fa9d3b766c2593acad01faedca89a`,
  `bcaa5b2531d422e7eabd09b92fc8f6659c44cbd879614d5d83ee2b9bbc24a736`,
  `a6736d2b12307cd6cde462513235b4c6f7517289ed89acfde8441832ebcce875`.
- Combined S0-R5/C1-R2A adjudication/report:
  `a8b41382b68488f393eafd9f057ddc5140ee8cb2c04ab3c3833a07345311f265`,
  `cd91e9226f99e1177caff83a62a438c29651af462a53d95891fdbc95b9477e06`.
- Official `Qwen/Qwen3-Coder-Next` revision
  `a19358a7659bd1f564300250ee189120c49a562f`, shard 1 byte count
  `3,999,619,288`, SHA-256
  `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- CAP0X-R2 coordinator/Intel/NVIDIA evidence:
  `d807e0867c41ba43e0ee86b2bbf6d14bba7db582d7084db390472739714f2a3d`,
  `3ded9dc2ea6e949cd50aca61796d966f809ea4dd2b318a7b705ef9388b099f95`,
  `10d6e6906218e5eaee33c5503dc26d8f0f9d344bf0a68f7ec2b24f8d13b1dd29`.
- CAP0X-R2 report/independent report:
  `23bbc3a7b4c8a824aea890073049d64e049019fea7719bde4143366c3edb9416`,
  `1af99bc75fe5ca7ba16892d632b809c8b308df5aba9827a9526584ff244bc105`.
- Reused mechanism sources: Intel host-USM
  `run_st2_mini_host_usm_q5.py`
  `399b79e819ec09e77ca5cde8683783b9f1a3c41d0cae5eef45f712585718aac2`;
  Intel width-8 `run_st2_mini_ergv_w8.py`
  `6472de274fa68a9f577b1483ef1225607f8425ac8587cb348d0c328cff7126ca`;
  NVIDIA D7 `run_port80b_d7_staged_exact_q5_plane.py`
  `26d4daba81d5f132857f9b584dfb12f3634874a2e9ee2290f9221c486ef4059a`.

CAP0X-R2 proved only that both backend processes completed correctly while
their process lifetimes overlapped. Its throughput and synthetic-payload claims
are not inherited. S0-R5 remains formally negative because one frozen natural
mutation was not output-observable; only its 96/96 numerical-quality arm and
C1-R2A's separate synthetic sensitivity control are positive.

## Frozen validation row and access seal

The only row scored is the final position, zero-based index 15, of D2-R3
`p0_whole`. Its rank-ordered expert IDs and native BF16 weights are:

| rank | expert | BF16 weight | BF16 word | owner |
|---:|---:|---:|---:|---|
| 0 | 50 | 0.2490234375 | 15999 | Intel |
| 1 | 199 | 0.14453125 | 15892 | Intel |
| 2 | 237 | 0.130859375 | 15878 | Intel |
| 3 | 474 | 0.126953125 | 15874 | Intel |
| 4 | 245 | 0.0810546875 | 15782 | NVIDIA |
| 5 | 374 | 0.0703125 | 15760 | NVIDIA |
| 6 | 239 | 0.057373046875 | 15723 | NVIDIA |
| 7 | 8 | 0.053955078125 | 15709 | NVIDIA |
| 8 | 168 | 0.047607421875 | 15683 | NVIDIA |
| 9 | 12 | 0.0380859375 | 15644 | NVIDIA |

The ten-ID byte digest is
`ea47c4b4b3b2942876101be4dc85072554805de8fef20d91ab531b64c731a462`;
the ten BF16-weight byte digest is
`249c79806e09cf86b0bd6aba465050621dea3163d362fffea5ace2f655e7c8a7`.
Rank determines device ownership only. It never determines accumulation order.

The D2 reader may materialize only these exact p0 tensors:

| key suffix | dtype/shape | SHA-256 |
|---|---|---|
| `post_norm` | BF16 `[1,16,2048]` | `d82286fac9616cdf8b03b8eddb8347acd3679afb639c8db696daf3f643084853` |
| `official_router_ids` | I64 `[16,10]` | `c183be31d947f3a74865eb58f874a0ffd2289adbe455d4689d38239c5a6be2ca` |
| `official_router_weights` | BF16 `[16,10]` | `d048f9eddc9f3e358d59383557da8f3fc3b91ab84baddb6b412c82164b2e3be2` |
| `experts` | BF16 `[16,2048]` | `a74a8a9ef47df5a43ff6ca3ecd28a14650c6275586b21ef6c0fc9f1c3559477c` |
| `shared` | BF16 `[16,2048]` | `3e1f0052460430ca03c19f7a312a80c68034d86b387d3981ae0cce3224e67125` |
| `shared_gate` | BF16 `[16,1]` | `3630e2b1cb0ad297f0efd2f029140f5befd810c3520c4dc7eeb0ce746ed49fc0` |

Parsing the common safetensors header is disclosed. Reading, mapping or hashing
any p1-p3 tensor value range is forbidden. A key-to-byte-range allowlist, logged
before access, rejects every nonlisted key and an independent verifier checks
that completed reads have zero intersection with all p1-p3 value ranges.

## Source tensors, exact Q5 and memory bound

Only 33 official BF16 projection tensors are legal: `gate_proj`, `up_proj` and
`down_proj` for the ten listed routed experts plus `shared_expert`. Gate/up are
`[512,2048]`; down is `[2048,512]`. Every source slice is read from the pinned
shard through a read-only mapping and independently hashed with key, shape,
dtype, offset and byte count before quantization.

Codec semantics are exactly S0-R5: row-major group 128; FP32
`max(abs(group))/15`; zero group uses BF16 scale `1` and all q zero; otherwise
round-to-nearest q clamped to `[-15,15]`; stored field `q+15` in `[0,30]`; field
31 forbidden; eight little-order 5-bit fields in five bytes; BF16 scales. For
each 1,048,576-element projection this is 655,360 code bytes plus 16,384 scale
bytes. The complete 33-projection compressed working set is therefore exactly
`22,167,552` bytes (`21.140625 MiB`), excluding small manifests. There is no
persistent bank. Quantization streams one source projection at a time and drops
the BF16 source view before the next projection; total coordinator plus child
record material must remain below 128 MiB and final retained evidence below
16 MiB.

The future independent verifier must reread and requantize all 33 allowed source
tensors without importing the builder or device codec, compare code bytes,
scale bytes, decoded BF16 bytes and canonical manifest digests, and prove field
31 absent.

## Shape-faithful component graph

PV0 must not turn the final row into a one-row GEMV. For each of the ten selected
experts it derives `token_index, topk_position = torch.where(p0_ids == expert)`
over the full 16-by-10 p0 route table, preserving the official full-sequence
gather shape and increasing-token order. Each device computes all occurrences
of each expert it owns and retains them; only the occurrence at token index 15
is scored in the final merge. This explicitly avoids the D2-R3
shape-dependent-BF16 confound.

The independent CPU source-BF16 graph and CPU Q5/ERGV graph share no quantizer,
decoder, reduction or merge helper with either child. For every expert:

1. BF16 input and BF16 source/Q5-decoded gate and up weights enter the frozen
   width-8 ERGV reduction order; gate and up endpoints are stored BF16.
2. `torch.nn.functional.silu(gate_bf16, inplace=False)` returns BF16, followed
   by BF16 `silu_bf16 * up_bf16`; gate, up, SiLU and activation are retained.
3. BF16 activation and BF16 down weights enter the same ERGV order; down is
   stored BF16 and retained.
4. For token 15, down is multiplied by that expert's captured native BF16 route
   weight and cast BF16.
5. Contributions are accumulated into a BF16 zero vector in ascending expert
   ID order `8,12,50,168,199,237,239,245,374,474`. Every accumulator state is
   retained. Route-rank order is a forbidden merge.
6. NVIDIA computes shared gate/up/SiLU/activation/down. The outer shared gate is
   the captured native BF16 linear value passed to `torch.sigmoid`; the returned
   BF16 value multiplies shared-raw in exact `sigmoid_gate * shared_raw` operand
   order. Shared raw and gated vectors are retained.

The source-BF16 routed and shared-raw full-16 arrays must be bitwise equal to the
retained D2-R3 `p0_whole_experts` and `p0_whole_shared` arrays before any device
result is eligible. Failure is `source_oracle_negative`, not retuned.

## Process-isolated device split

The coordinator is the only process allowed to open D2 or the official shard.
It launches two fresh, separately identified child processes through the
CAP0X-R2 process-isolation pattern. Each child records PID, parent PID, process
creation time, executable/argv, nonce, device identity, backend/driver/compiler
identity, ready/submit/complete/exit QPC and cleanup status. The children never
open D2, the checkpoint, the other child's package or the CPU oracle arrays.

- Intel receives only ranks 0-3 and uses the proven OpenCL Intel host-USM class:
  Q5 records and inputs live in `clHostMemAllocINTEL`, pointer arguments use
  `clSetKernelArgMemPointerINTEL`, and no `cl_mem`, write, copy or migrate API may
  materialize a hidden weight copy.
- NVIDIA receives ranks 4-9 plus shared and uses the frozen D7/Next width-8
  staged-Q5 CUDA path. Its pointer/copy ledger must bind every small staging
  allocation and prove that no unlisted weight source is used.

Both children must be alive at the release barrier and must complete their own
device work and explicit cleanup with exit code zero. Kernel overlap is logged
but is neither required nor claimed. There are no repeats and no latency,
bandwidth, percentile, ratio or speedup measurements.

## Numerical gates

All metrics use FP64 accumulation over the retained BF16 words. For reference
`a` and candidate `z`, `relL2 = ||z-a||2 / ||a||2`; if both norms are zero it is
zero, and if only the reference norm is zero it is infinity. `max_abs`, cosine,
different-BF16-word count and signed-order BF16 max ULP are retained; only gates
named below adjudicate.

1. Decoded Q5 weights and every gate/up endpoint from each device must be
   bitwise equal to its independently produced CPU-Q5/ERGV endpoint.
2. Because device transcendental implementations can differ, each device
   SiLU, activation and down array may be non-bitwise only if it is finite and
   has `relL2 <= 0.001` against the CPU-Q5/ERGV array. No alternate threshold or
   backend-specific threshold is permitted.
3. The merged heterogeneous routed, shared-raw and shared-gated token-15 vectors
   must each have `relL2 <= 0.001` against the CPU-Q5 component vectors.
4. Independently, CPU-Q5 and heterogeneous routed, shared-raw and shared-gated
   token-15 vectors must each have `relL2 <= 0.08` against the source-BF16/D2
   component vectors. `0.08` is inherited unchanged from S0-R5; S0-R5 observed
   maxima were routed `0.068281357623850403`, shared-raw
   `0.0750221523726263`, shared-gated `0.075311459958494031` across its broader
   numerical arm. These prior values motivate but do not guarantee PV0.
5. All retained source, oracle, child and merged arrays and all metric scalars
   must be finite. Exact dtype, shape, byte count and SHA-256 manifests are hard
   gates.

Thus Q5 quality and heterogeneous implementation error are separate. The loose
quality envelope cannot hide a backend error larger than `0.001`.

## Preregistered controls

Every safe checker records `(call_id, requested metadata, presented metadata,
codes/scales digest, queue counter before/after, verdict)` and must reject before
decode or enqueue:

1. wrong expert ID with otherwise valid shape;
2. gate/up projection swap;
3. wrong owner/rank slot across the Intel/NVIDIA boundary;
4. one changed packed 5-bit field with unchanged claimed digest;
5. field 31 injection;
6. any p1-p3 D2 value-key request.

Unsafe arithmetic is not used to decide whether a metadata control is valid.
For sensitivity, each backend separately uses its lexicographically first owned
down-projection field with source `q != 0`; mutate one step toward zero, choose
the first `k` in `[-8,8]` for a one-hot BF16 activation `2**k` that changes the
CPU BF16 output, then run the same fixed vector through the unsafe device path.
The checker must reject the mutated digest before unsafe execution, and the
unsafe output must change the preregistered BF16 word. If no such witness exists
or a device does not reproduce it, `control_negative`. This is a synthetic
integrity control, explicitly separate from natural p0 quality, and mirrors the
C1-R2A lesson without reinterpreting S0-R5.

The independent verifier rebuilds every requested/presented record, control
mutation, expected word and call ordering from raw source and raw arrays; stored
control booleans are never trusted.

## Resources, evidence and outcomes

Start available RAM must be at least 2 GiB; final available reserve at least
2 GiB. Process peak working set is capped at 1 GiB, Intel allocations at 128
MiB, NVIDIA allocations at 256 MiB and retained evidence at 16 MiB. The exact
22,167,552-byte Q5 working set is destroyed after independent-result
serialization; pre/post resource and cleanup ledgers must balance. Any hidden
full-weight copy, p1-p3 value read, unlisted source tensor, surviving child,
unreleased device allocation or retry invalidates the attempt.

Adjudication is exactly one of:

- `real_weight_process_validation_positive`: every source, codec, device,
  merge, quality, control, seal, process and cleanup gate passes;
- `source_oracle_negative`, `codec_negative`, `device_numerical_negative`,
  `quality_negative`, `control_negative`, `invalid_protocol`;
- `blocked_capability` or `blocked_resource`, only when the named prerequisite
  fails before adjudicable device output.

One clean physical attempt is permitted only after a future implementation,
independent source audit, static preflight and separate explicit execution GO.
No retry or threshold change is authorized by this document.
