# HET-NEXT-L0-PV0-R1 — selected-expert real-weight process validation

## Status, supersession and claim

This immutable design preregistration supersedes PV0 only for future work. PV0
and its independent NO-GO audit remain negative evidence. PV0-R1 authorizes no
runner, preflight, payload read, shard scan, quantization, compiler or device
call. Implementation requires a new independent design GO.

The only possible positive claim is:

> On the already known official Qwen3-Coder-Next layer-0 `p0/n16` row, two
> process-isolated device workers reproduced an independent CPU Q5/ERVG oracle
> for a fixed split of the ten naturally selected real experts plus shared, and
> their official-order token-15 component aggregate met the frozen numerical
> envelope against the official BF16 capture.

This is validation-only and non-heldout. It is not full-p0 reconstruction,
layer/logit equality, model quality, performance, kernel-overlap, acceleration,
deployment, novelty or a breakthrough.

## Exact evidence bindings

Future source/locks bind path, bytes and SHA-256 for:

- D2-R3 raw/result/audit/interpretation:
  `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`,
  `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`,
  `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`,
  `be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0`;
- S0-R5 raw/result/commit and C1-R2A raw/result/commit:
  `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`,
  `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91`,
  `d784ded5e7893095e2f27b75695e635c9cc880109736c87496138e3188509372`,
  `d7272ce6aec3533b487829360e40398f1d5fa9d3b766c2593acad01faedca89a`,
  `bcaa5b2531d422e7eabd09b92fc8f6659c44cbd879614d5d83ee2b9bbc24a736`,
  `a6736d2b12307cd6cde462513235b4c6f7517289ed89acfde8441832ebcce875`;
- combined S0/C1 adjudication/report:
  `a8b41382b68488f393eafd9f057ddc5140ee8cb2c04ab3c3833a07345311f265`,
  `cd91e9226f99e1177caff83a62a438c29651af462a53d95891fdbc95b9477e06`;
- official revision `a19358a7659bd1f564300250ee189120c49a562f`, shard 1
  `3,999,619,288` bytes and SHA-256
  `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`;
- CAP0X-R2 result/Intel/NVIDIA and independent report:
  `d807e0867c41ba43e0ee86b2bbf6d14bba7db582d7084db390472739714f2a3d`,
  `3ded9dc2ea6e949cd50aca61796d966f809ea4dd2b318a7b705ef9388b099f95`,
  `10d6e6906218e5eaee33c5503dc26d8f0f9d344bf0a68f7ec2b24f8d13b1dd29`,
  `1af99bc75fe5ca7ba16892d632b809c8b308df5aba9827a9526584ff244bc105`;
- Intel host-USM/width-8 and NVIDIA D7 mechanism sources:
  `399b79e819ec09e77ca5cde8683783b9f1a3c41d0cae5eef45f712585718aac2`,
  `6472de274fa68a9f577b1483ef1225607f8425ac8587cb348d0c328cff7126ca`,
  `26d4daba81d5f132857f9b584dfb12f3634874a2e9ee2290f9221c486ef4059a`.

CAP0X-R2 supplies backend feasibility only, not a formal process harness or a
performance inheritance. S0-R5 remains formally control-negative; only its
numerical-quality subarm and C1's separate synthetic sensitivity control are
usable context.

## Header-only D2 seal

The future reader first reads exactly the 8-byte safetensors header length plus
the 170,656-byte JSON header, total 170,664 bytes, SHA-256
`8eed6e625a1cac3e0cf71e621d95fa901d7f2ff517e7d307c24435b1baa3c2f4`.
It never calls a generic safetensors opener and never maps the D2 file. It
validates the 171,696,126-byte file size and constructs a header-only
key-to-absolute-range manifest. It may then seek/read only these p0 ranges:

| key suffix | absolute half-open range | dtype/shape | payload SHA-256 |
|---|---|---|---|
| `official_router_ids` | `[216552,217832)` | I64 `[16,10]` | `c183be31d947f3a74865eb58f874a0ffd2289adbe455d4689d38239c5a6be2ca` |
| `experts` | `[154798500,154864036)` | BF16 `[16,2048]` | `a74a8a9ef47df5a43ff6ca3ecd28a14650c6275586b21ef6c0fc9f1c3559477c` |
| `official_router_weights` | `[155077028,155077348)` | BF16 `[16,10]` | `d048f9eddc9f3e358d59383557da8f3fc3b91ab84baddb6b412c82164b2e3be2` |
| `post_norm` | `[155077348,155142884)` | BF16 `[1,16,2048]` | `d82286fac9616cdf8b03b8eddb8347acd3679afb639c8db696daf3f643084853` |
| `shared` | `[155143556,155209092)` | BF16 `[16,2048]` | `3e1f0052460430ca03c19f7a312a80c68034d86b387d3981ae0cce3224e67125` |
| `shared_gate` | `[155209092,155209124)` | BF16 `[16,1]` | `3630e2b1cb0ad297f0efd2f029140f5befd810c3520c4dc7eeb0ce746ed49fc0` |

Every read is checked against an immutable internal allowlist and logged before
and after. Any other key, range, overlap, whole-file hash attempt or mapping is
`invalid_protocol`. P1-p3 value ranges are never opened, hashed or copied by
builder, children or verifier. The full D2 SHA above is provenance from prior
evidence, not recomputed during PV0-R1.

## Frozen row, split and selected hit shapes

Token 15 route IDs are rank-order
`[50,199,237,474,245,374,239,8,168,12]`, digest
`ea47c4b4b3b2942876101be4dc85072554805de8fef20d91ab531b64c731a462`.
Their native BF16 words are
`[15999,15892,15878,15874,15782,15760,15723,15709,15683,15644]`,
digest `249c79806e09cf86b0bd6aba465050621dea3163d362fffea5ace2f655e7c8a7`.
Intel owns ranks 0-3; NVIDIA owns ranks 4-9 plus shared.

PV0-R1 computes only these ten experts, but preserves their official p0
full-sequence gather shapes. Define `mask=one_hot(ids,512).permute(2,1,0)` and,
for each selected expert in ascending ID order, exactly
`topk_position,token_index=torch.where(mask[expert])`. The frozen occurrences
are:

| expert | token indices in returned order | top-k positions | hits |
|---:|---|---|---:|
| 8 | `15,13` | `7,8` | 2 |
| 12 | `13,15` | `3,9` | 2 |
| 50 | `15,13` | `0,7` | 2 |
| 168 | `8,15` | `1,8` | 2 |
| 199 | `13,15` | `1,1` | 2 |
| 237 | `13,15` | `2,2` | 2 |
| 239 | `8,15` | `6,6` | 2 |
| 245 | `13,15` | `4,4` | 2 |
| 374 | `15` | `5` | 1 |
| 474 | `5,8,13,15,0,11` | `0,0,0,3,6,8` | 6 |

Total routed selected hits are exactly 23: Intel 12 and NVIDIA 11. Other p0
experts and their 30+ tensors are forbidden. At token 15 the selected ten are
the complete official top-10 set, so no routed contribution is omitted from
that token's aggregate; no such assertion is made for tokens 0-14.

## Source and Q5 oracles

Only 33 shard tensors may be read: gate/up/down for the ten experts and shared.
Gate/up shapes are `[512,2048]`, down `[2048,512]`. Source reads are read-only,
offset-bound, one projection at a time. Q5 is exactly S0-R5: group128, FP32
maxabs/15, zero group BF16 scale 1/q0, q in `[-15,15]`, stored q+15 in
`[0,30]`, field31 forbidden, eight little-order fields/five bytes and BF16
scales. Codes plus scales are 671,744 bytes/projection, 2,015,232/triplet and
exactly 22,167,552 bytes for 33 projections. There is no persistent bank.

The graph-matched source reference uses official PyTorch structure, not ERGV:
for each selected expert and the frozen `torch.where` order, concatenate source
BF16 gate then up weights to `[1024,2048]`, call one fused `F.linear`, split
gate-then-up, apply BF16 `F.silu(gate)*up`, call BF16 down `F.linear`, multiply
native BF16 route weights, and `index_add_` into BF16 zeros in ascending expert
ID order. Shared uses the same fused gate/up and exact
`torch.sigmoid(shared_gate_bf16) * shared_raw_bf16` operand order.

This source graph is **not** required bitwise equal to the official grouped MoE.
Its selected token-15 routed vector is compared with D2 `experts[15]`; shared raw
with D2 `shared[15]`. Both must be finite and satisfy max-absolute error
`<=2.44140625e-4` and relative-L2 `<=1.2754377e-4`. These are explicitly the
published worst D2-R3 shape-dependent diagnostic bounds and are used only as a
conservative validation gate, not claimed as an official exact oracle. Failure
is a valid `source_graph_negative`; there is no retune.

The independent CPU Q5/ERVG oracle shares no quantizer, decoder or reduction
helper with the device workers. It consumes decoded BF16 weights and freezes
width-8 reduction order. Gate/up are stored BF16; BF16 SiLU and activation feed
BF16 down. Token-15 weighted contributions accumulate in ascending expert-ID
order `8,12,50,168,199,237,239,245,374,474`, retaining every state.

## Device contracts and numerical gates

Intel receives only its 8,060,928 Q5 record bytes. Every logical Intel buffer is
one `clHostMemAllocINTEL` allocation: records 8,060,928; input 65,536; gate
12,288; up 12,288; SiLU 12,288; activation 12,288; down 49,152; token-15
weighted outputs 16,384 bytes. Declared payload total is 8,241,152 bytes before
alignment. All kernel pointers use `clSetKernelArgMemPointerINTEL`; output is
read directly by CPU from host-USM after `clFinish`. Weight/input/output use zero
`clCreateBuffer`, enqueue-write/read/copy/migrate calls. Any implementation-only
scratch allocation must be separately declared, host-USM, below 1 MiB and
cannot contain dense weights.

NVIDIA receives 14,106,624 Q5 record bytes for six routed experts plus shared.
Its D7/Next width-8 path uses only enumerated pinned-host and device staging
allocations; device-produced activation directly feeds down. Neither child may
read D2, shard, CPU-oracle arrays or the other package.

Hard numerical gates are:

1. independent requantization reproduces all codes/scales/decoded BF16 bytes;
2. each device gate and up BF16 endpoint is bitwise equal to CPU-Q5/ERVG;
3. device SiLU, activation and down arrays are finite and each has
   `relL2<=0.001` versus CPU-Q5/ERVG; exact equality is retained diagnostically;
4. heterogeneous token-15 routed/shared raw/shared gated each has
   `relL2<=0.001` versus CPU-Q5/ERVG;
5. CPU-Q5 and heterogeneous token-15 routed/shared raw/shared gated each has
   `relL2<=0.08` versus its graph-matched source-BF16 counterpart, unchanged
   from S0-R5.

The blanket word “exact” never applies downstream of a non-bitwise SiLU. Down
and merge use the `0.001` implementation envelope. Q5 quality and device error
are separately adjudicated.

## Standalone process protocol

CAP0X code is not the formal harness. The future coordinator uses a standalone
protocol: create Intel and NVIDIA suspended, put both in a fresh Windows Job
Object with `KILL_ON_JOB_CLOSE`, then resume. Anonymous pipes use inheritable
child ends only; messages are 8-byte little-endian length (`<=1 MiB`) plus
canonical JSON containing protocol revision, nonce, role, strictly increasing
sequence and schema. Raw tensors travel only through the prehashed read-only
child packages/result files, never JSON.

Both children must emit `ready` within 300 seconds. After both ready messages,
one nonce-bound `start` frame releases them. Each emits one result frame within
1,800 seconds, cleans its backend, exits zero within 30 seconds, and leaves no
live PID with the recorded creation time. Every read uses PeekNamedPipe/ReadFile
with QPC deadlines and exit-code checks; partial/oversize/wrong-nonce frames
fail. On error, coordinator sends abort, terminates the job if needed, performs
bounded waits, closes every handle and writes immutable failure evidence.

Raw/result candidates are create-new temporaries. A separately launched
independent verifier must return a hash-bound positive artifact before the
coordinator promotes result files; commit is fsynced and promoted last. Valid
existing commit returns `already_complete`; stale artifacts are quarantined and
abort without a physical retry. Process overlap is evidence, not a gate or
performance claim.

## Exact controls

Safe metadata controls reject before decode/enqueue: wrong expert, gate/up swap,
wrong owner/rank, changed code digest, field31 and any nonallowlisted D2 range.
Queue counters must remain unchanged. The verifier reconstructs metadata and
does not trust booleans.

Each backend has one deterministic sensitivity witness chosen before any device
call. Enumerate assigned **down** records by expert ID, then row, then column.
Choose the first field with source q nonzero and one-step-toward-zero
`q'=q-sign(q)` still in `[-15,15]`. The input activation is one-hot at that
column. Scan integer `k=-8..8`; choose the first BF16 amplitude `2**k` for which
the independent CPU-Q5 reduction changes the selected output-row BF16 word.
Freeze in the CPU manifest: expert, projection, row, column, group, slot, source
q, q-prime, scale word, k, activation SHA, original/mutated code digests and
expected original/mutated output words/XOR. The safe checker must reject the
mutated digest; only then may the unsafe child path reproduce the two words.
No natural-output observability is required or inferred.

## Exact evidence sizing, resources and outcomes

Selected routed stage payload per implementation is: four `[23,512]` BF16
gate/up/SiLU/activation arrays = 94,208 bytes; `[23,2048]` down = 94,208;
ten weighted token-15 vectors plus ten accumulator states = 81,920; total
270,336 bytes. Source CPU, Q5 CPU and two device partitions together are bounded
by three logical copies, 811,008 bytes. Shared full-16 gate/up/SiLU/activation,
raw and gated plus outer-gate total 196,640 bytes per implementation, 589,920
for three. Inputs/routes/control tensors are <256 KiB. Therefore normative raw
tensors are <1.7 MiB; allowing safetensors header, compiler logs, manifests,
process transcripts and failure evidence, the final artifact hard cap is 8 MiB.

Start/final available RAM must be >=2 GiB; aggregate process peak working set
<=1 GiB; Intel allocation <=16 MiB; NVIDIA allocation <=64 MiB. Source/Q5
packages are destroyed after verified serialization; releases and child exits
must balance. One clean attempt, no retry.

Outcomes: `real_weight_process_validation_positive`,
`source_graph_negative`, `codec_negative`, `device_numerical_negative`,
`quality_negative`, `control_negative`, `invalid_protocol`,
`blocked_capability`, or `blocked_resource`.
