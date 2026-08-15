# HET-NEXT-L0-PH1-R1 — CPU evidence and exact physical contract

Date: 2026-08-14  
State: immutable design revision; device implementation remains closed pending independent CPU-artifact and R1-design audits.

This revision binds the completed CPU-only PH1 package and supersedes only the underspecified LUT, BF16-multiply, source-stage, device-ledger and control-cardinality passages of PH1 preregistration SHA-256 `c464be6643f0301ea9f99b0e69141959a53667fa7cf9915bd540cea0a15b2b39` and design SHA-256 `4fa8a9f17b5d6c16d92c6ff1816ceda7e213e852e7fac5bfe5761c3c0338bbaf`. All inputs, records, quality threshold, claim limits and sequential Intel-before-NVIDIA order remain unchanged.

## Committed CPU package

Package: `reports/streamq5_moe/het_next_l0_ph1_cpu_freeze_r2`.

| file | bytes | SHA-256 |
|---|---:|---|
| `bf16_silu_lut.bin` | 131,072 | `a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed` |
| `high_precision_silu_diagnostic.bin` | 131,072 | `f2efcbdc3b94b42a24dfe187321ae2a426e7685ab447e05452be994e843693c2` |
| `cpu_stage_freeze.safetensors` | 23,432 | `c2fbc4d6c3c400ecb0ac7af36b36c88a1c8122d3066cb123430f934bd750d6a8` |
| `cpu_stage_freeze.json` | 4,054 | `520b19d320cf88c71c5c972d0cb3b7ad8b5e29ed152f155909daba9e1d442090` |
| `handoff.json` | 2,317 | `281d06d367b2fde359333bc5f3f8e646171e27111341ca1719e62f8d595086b9` |
| `manifest.json` | 670 | `63f6c842f377fb18738d6016b133c7529803581d0cd661739c0ffd648a82ac54` |
| `commit.json` | 293 | `f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06` |

The commit, exact package file set, manifest, hashes, R2 authorization and resource evidence must independently pass before device sources may be frozen. CPU computation used 659,292,160-byte peak working set; host available was 49,633,964,032 bytes at start and 49,655,648,256 before commit; no device or compiler opened.

CPU source versus Q5 down output has row-major FP64 `rel_l2 = 0.040058847132189`, `max_abs = 0.00244140625`, and 1,970 differing BF16 words out of 2,048. The preregistered quality gate `rel_l2 <= 0.08` is therefore positive. This result cannot be retuned or replaced.

## Normative SiLU truth and independent diagnostic

The canonical 131,072-byte LUT file above is the normative PH1 activation contract because it records the exact official pinned PyTorch BF16 `F.silu` endpoint used by the source graph. Device code and the exact CPU-Q5 oracle consume those bytes; they never call an exponential implementation. The LUT generator is the immutable scientific freezer SHA-256 `746a879192041dee32acb1bcb9360ce9dde6775631c0a0671312660fb71437c8`, executed under the runtime and dependency locks recorded in the committed handoff.

The high-precision diagnostic is independently generated with mpmath 1.3.0 at 100 decimal digits from `x/(1+exp(-x))`, then rounded directly to BF16 ties-to-even. It is not normative and may not replace the PyTorch mapping. Its SHA-256 is `f2efcbdc...`; it differs from the official LUT in exactly 145 of 65,536 entries, all retained as evidence. The independent verifier must (a) hash and structurally validate the normative artifact; (b) independently reconstruct the 100-digit diagnostic and exactly reproduce its digest and 145-word relation; and (c) prove that each of the 512 actual gate words indexes the retained normative value. This is deliberately non-circular: the artifact defines official implementation truth, while the separately implemented formula audit explains its mathematical relationship.

## Exact BF16 multiplication

For finite BF16 words `a,b`, widen each by placing the 16-bit word in FP32 bits 31..16. Reject if either exponent field is 255. If either magnitude is zero, return a signed BF16 zero whose sign is `sign(a) XOR sign(b)`. Otherwise unpack each FP32 operand as signed integer significand `n` and power-of-two exponent `e`, multiply the integer significands exactly, and normalize the exact integer product. For a normal result retain 24 significand bits with round-to-nearest ties-to-even, carrying into the exponent when necessary. For a subnormal result round the exact magnitude to units of `2^-149`; signed underflow becomes signed zero. Overflow to infinity is invalid. Finally round the finite FP32 endpoint once to BF16 RNE using `((bits + 0x7fff + ((bits>>16)&1))>>16)`. No host/device floating multiply, FTZ/DAZ, reassociation or double rounding is permitted.

Normative test vectors `(a,b)->result`, in raw hex BF16 words, are:

- `(0000,3f80)->0000`; `(8000,3f80)->8000`; `(0000,bf80)->8000`; `(8000,bf80)->0000`;
- `(3f80,3f80)->3f80`; `(bf80,3f80)->bf80`; `(3f00,4000)->3f80`;
- `(0001,3f80)->0001`; `(0001,3f00)->0000`; `(0003,3f00)->0002`;
- `(7f7f,0001)->3cff` and must never flush either input;
- any NaN or infinity operand must reject before multiplication.

The 512 actual activation words have frozen SHA-256 `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f`. Independent implementations must reproduce every word.

## Frozen source and Q5 stages

The official source graph environment is Python 3.12.10, Torch 2.12.1+cu132, AVX2, threads/inter-op 1, deterministic algorithms true, highest matmul precision, MKLDNN true, flush-denormal false, inference mode true and autocast false; CPU identity is `Intel64 Family 6 Model 197 Stepping 2, GenuineIntel`, affinity 0..15. Qwen3-Next modeling source SHA-256 is `de40823607becdd616436e3b332f14e0c92df5495ac72ef8af027c4488b9afca`; Transformers activations source SHA-256 is `5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665`.

Every stage is BF16, finite and has this immutable SHA-256:

| stage | shape | SHA-256 |
|---|---:|---|
| natural input | `[2048]` | `5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f` |
| source fused gate_up | `[1024]` | `94550e9b214edd4713aff00902ee5083f0bf1d9e633bf43a950ecdcf5f8efdf7` |
| source gate | `[512]` | `2a898f7c33c8df8ed441222cfe3a62672fab0e5ae612905e0bf98cd53ea861cc` |
| source up | `[512]` | `0018b298d0c0f55fa38a8fd0141fb4684601911cfd88e7da4cf2480083cf580f` |
| source SiLU | `[512]` | `184fb8cc8c0a46cd7a6f00c65350d8cd12e3a38defb899e5401caf2d3f2d03be` |
| source activation | `[512]` | `598a656ed0d56ae51bd503ffcdb93f73fff239ab725000209469835b08dbfa26` |
| source down | `[2048]` | `ed49c260c3b09985dbfec10106a04eaea99b59d97114514f7099d4bdb84c6e09` |
| CPU-Q5 gate | `[512]` | `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867` |
| CPU-Q5 up | `[512]` | `f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08` |
| CPU-Q5 SiLU | `[512]` | `a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8` |
| CPU-Q5 activation | `[512]` | `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f` |
| CPU-Q5 down | `[2048]` | `142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc` |

## Exact Intel physical ledger

Before `OpenCL.dll` or any OpenCL call, the Intel child independently rechecks the opened authorization lock, package commit, all source hashes and a CPU-eligibility PASS hash. It then emits one totally ordered ledger with these mandatory rows:

1. identity/enumeration and extension proof for exactly the frozen Intel Arc target;
2. context creation, one in-order queue creation, program creation, program build, binary retrieval;
3. four kernel creations: `gate_linear`, `up_linear`, `activation`, `down_linear`;
4. fourteen host-USM allocation rows in the exact buffer-table order from the base PH1 design. Each row records requested bytes/alignment4096, nonzero unique pointer, pointer modulo4096, and independent `clGetMemAllocInfoINTEL` allocation-type=host/base=pointer/size=bytes queries;
5. nine CPU-direct initialization rows: five stage outputs to `0xffff`, four counter arrays to uint32 zero;
6. eighteen `clSetKernelArgMemPointerINTEL` rows: gate4, up4, activation6, down4, with exact named pointer mapping;
7. four enqueue rows, gate/up global4096 local256, activation global512 local256, down global16384 local256; no event object requested;
8. one `clFinish` row;
9. CPU-direct read evidence for five stage buffers and four counter buffers, after finish only;
10. reverse release-attempt rows for fourteen USM allocations, four kernels, program, queue and context: exactly 21 release attempts even after an earlier release error;
11. final cleanup row with all return codes and zero live owned resources.

Forbidden-call counters for every `cl_mem`, buffer, enqueue read/write/copy, migrate and prefetch API must each be zero. The retained OpenCL source, exact options, build log and program binary have predeclared hashes in the implementation freeze; source audit checks absence of fast-relaxed math, unsafe contraction and FTZ. A PH1 Intel positive additionally requires all five raw stage hashes equal the CPU-Q5 table, all four counters all-one and all canaries overwritten.

## Exact NVIDIA physical ledger

Before CUDA/CuPy/driver access, the NVIDIA child independently rechecks the opened authorization, CPU eligibility, committed Intel PASS and clean-release hashes. It uses a direct precompiled sm_120 cubin with two entry points (`q5_linear`, `bf16_lut_activation`), loaded through the CUDA Driver API; no source compile or textual PTX path is allowed.

The totally ordered success ledger is:

1. exact identity/enumeration, driver/runtime and distinct-PCI proof;
2. primary-context observation, one nondefault stream creation, module load, two function lookups;
3. fourteen pinned-host allocations in exact buffer-table order, then fourteen device allocations in that order; every pointer is nonzero and unique within its class;
4. nine device memsets in stage/counter order;
5. five H2D rows: gate record675840, up record675840, down record675840, input4096, LUT131072;
6. four kernel rows: gate grid16/block256, up16/256, activation2/256, down64/256, each on the named stream with stable uint64 pointer arguments;
7. nine D2H rows: gate1024, up1024, SiLU1024, activation1024, down4096, gate/up/activation counters2048 each, down counters8192;
8. one stream synchronization;
9. reverse release attempts for fourteen device allocations, fourteen pinned allocations, module and stream: exactly 30 release attempts even after an earlier error;
10. final cleanup row with zero live owned resources. The CUDA primary context is observed but not owned or reset by PH1.

The implementation freeze binds CUDA source, direct NVRTC version/options/log, PTX, cubin and disassembler tool/output hashes. Static audit must prove sm_120, exact two entrypoints, exact launch contract, width-8 DAG, no `.ftz`, no fast-math/reassociation, and direct module loading. Mutation tests change one ledger row count/direction/bytes, pointer identity, device identity, output word, counter and cleanup code; every change must fail.

## Controls and next authorization

There are exactly 22 predevice safe controls: seven per record (truncation; wrong projection ordinal; stale CRC; valid-CRC code mutation; scale mutation; recomputed-CRC field31; wrong input) and one global wrong-LUT digest. Each records the exact expected checker stage and proves compile/alloc/launch counters remain zero. The three outcome-independent q-step numerical witnesses remain separate unsafe CPU diagnostics and do not add safe-control rows.

Only after (a) the completed CPU package independently passes, and (b) this R1 design independently passes, may standalone sources be implemented. Implementation still requires a new source audit, then a no-device static preflight, then a separate one-attempt authorization. No device action is authorized by this document.
