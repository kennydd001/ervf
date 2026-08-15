# HET-NEXT-L0 PH1 NVIDIA full-expert N0 preregistration

Date: 2026-08-14. State: design-only, execution closed. No implementation, preflight, compiler, CUDA library load or device action is authorized by this document.

## Narrow objective and immutable prior evidence

N0 asks one question: can the pinned NVIDIA GPU reproduce, bit for bit, the already frozen CPU-Q5 gate, up, normative-LUT SiLU, exact BF16 activation and down stages for one official real expert-50 MLP on the exact D2R3 `p0_n16` natural input used by the positive Intel execution?

The immutable Intel final verification is `het_next_l0_ph1_intel_execution_r8v1r1a_independent_verification.json`, SHA-256 `42cd69582a47b8b5f8f4b7f24a696f1d3fcc6fbd49c05d0f61354a57cefc052d`, with `pass=true`, `bundle_adjudication=positive`, 10/10 current checks, 20/20 numerical checks, all 18 physical gates and exact cleanup. The bound R8A5 bundle is:

| file | bytes | SHA-256 |
|---|---:|---|
| `result.json` | 99,483 | `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50` |
| `manifest.json` | 167 | `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b` |
| `commit.json` | 210 | `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965` |

The CPU package commit is SHA-256 `f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06`; manifest `63f6c842f377fb18738d6016b133c7529803581d0cd661739c0ffd648a82ac54`; stage safetensors `c2fbc4d6c3c400ecb0ac7af36b36c88a1c8122d3066cb123430f934bd750d6a8`; normative LUT `a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed`.

N0 incorporates the independently GO-audited primary-context repair `HET_NEXT_L0_PH1_R2_NVIDIA_CONTEXT_CONTRACT_2026-08-14.md`, SHA-256 `dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c`, and audit SHA-256 `db5c6055bbffec2ea1e38e50f3b1c4d5dbece2deaf74c7ce5110056d05c8f1fe`. It also binds the PH0X-R12 direct no-FTZ cubin validation: prereg SHA `0a68cc7edf9acefba70f8f3a067ebcd207693a332ee23bc2d1d147b612d0906d`, independent verifier SHA `7ee3161f03b79be6475fec9ddc4936a28019632640e81b02ed18ed3b132e1b9b`, report SHA `6904cbc34eb935e55e455c8f006e75d0dbc6b1c46247a5c3fbb445a4001e344a`, and immutable physical result SHA `159d77b8fc6d1cac3d2123c09b7f256837c9691a397fa9d052309752d26955bc`.

No Intel or NVIDIA result is rerun or overwritten. N0 has one fresh namespace and one physical attempt after future source, compile, static-preflight and authorization audits.

## Exact data and arithmetic

The standalone CPU preparation rereads only the three frozen official shard ranges and the D2 input slice; it performs no model forward. It rebuilds the exact STREAMQ5 records in RAM using FP32 max-abs/15, ties-to-even rounding, clamp `[-15,15]`, stored field `q+15` in `[0,30]`, BF16 scale, group 128, eight little-order fields per five bytes, exact zero-group semantics, CRC and 64-byte header. Before CUDA is loaded, the safe checker requires every field at most 30, exact identity/input/digests and these artifacts:

| input | bytes | SHA-256 |
|---|---:|---|
| gate record, projection ordinal 0, `[512,2048]` | 675,840 | `e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9` |
| up record, ordinal 1, `[512,2048]` | 675,840 | `6da7025af27de06c4f6011ddfc82672263b6f0593b2dcacf77705a443f44fbfb` |
| down record, ordinal 2, `[2048,512]` | 675,840 | `bd1a8ef9ae689fefebf73408f3985c96a0725670dc0b0f7f46268a5a89d12157` |
| natural BF16 input `[2048]` | 4,096 | `5ce66a20ed658860ab4e98499e76205775cf0dd32cef15f35723dd83fc13fd3f` |
| normative BF16 SiLU LUT `[65536]` | 131,072 | `a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed` |

`q5_linear` has four pointer arguments `(record,input,output,counters)` and reads the frozen header. Headers with `(rows,cols)` other than `(512,2048)` or `(2048,512)` are non-dispatchable. For `[512,2048]`, each `cooperative_groups::tiled_partition<8>` tile owns one row; lane `l` evaluates packs `l+8v`, `v=0..31`; each pack evaluates its eight columns in increasing field order with `fmaf(round_bf16(float(q)*scale), bf16(input), accumulator)`. Per-lane partials reduce at distances 16,8,4,2,1 using `__fadd_rn`; the tile reduces with `shfl_down` distances 4,2,1 and `__fadd_rn`. For `[2048,512]`, `v=0..7` and the per-lane distances are 4,2,1; tile reduction is unchanged. Lane zero rounds FP32 once to BF16 ties-to-even and increments exactly its row counter once. There is no atomics-based output accumulation.

`bf16_lut_activation(gate,up,lut,silu,activation,counters)` uses one thread per row. It rejects exponent 255, indexes the normative LUT by the raw gate BF16 word, writes that raw SiLU word, computes `SiLU_word * up_word` with the frozen integer exact-BF16 multiply (including signed zero and subnormal behavior), writes activation, and increments its counter exactly once. It never calls `exp`, a floating multiply or a device SiLU intrinsic.

The exact expected finite BF16 stage outputs are:

| stage | words/bytes | SHA-256 |
|---|---:|---|
| gate | 512 / 1,024 | `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867` |
| up | 512 / 1,024 | `f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08` |
| SiLU | 512 / 1,024 | `a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8` |
| activation | 512 / 1,024 | `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f` |
| down | 2,048 / 4,096 | `142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc` |

All five byte arrays must equal the independent CPU oracle and immutable Intel arrays exactly. The four counter arrays contain exactly 512,512,512,2048 little-endian uint32 ones. All output buffers start as `0xff`, all counter buffers as zero, and exact output/counter equality plus finite-word checks are hard gates. The inherited CPU source-versus-Q5 down quality remains `rel_l2=0.040058847132189 <= 0.08`; N0 neither recalculates a threshold nor uses device output to tune it.

## Exact host and device buffers

The owner thread allocates fourteen pinned-host buffers with `cuMemHostAlloc` and fourteen device buffers with `cuMemAlloc_v2`, in this exact order and byte count:

| index | name | bytes |
|---:|---|---:|
| 0 | gate_record | 675,840 |
| 1 | up_record | 675,840 |
| 2 | down_record | 675,840 |
| 3 | natural_input | 4,096 |
| 4 | silu_lut | 131,072 |
| 5 | gate | 1,024 |
| 6 | up | 1,024 |
| 7 | silu | 1,024 |
| 8 | activation | 1,024 |
| 9 | down | 4,096 |
| 10 | gate_counters | 2,048 |
| 11 | up_counters | 2,048 |
| 12 | activation_counters | 2,048 |
| 13 | down_counters | 8,192 |

Each class totals exactly 2,185,216 bytes; combined requested pinned plus device storage is 4,370,432 bytes. Every pointer is nonzero and unique within its class. All Driver calls use 64-bit `CUdeviceptr` values and stable host variables for kernel-parameter pointer arrays.

Exactly five `cuMemcpyHtoDAsync_v2` calls copy indices 0..4 on the named nondefault stream, in order, for 675840,675840,675840,4096,131072 bytes. Exactly nine `cuMemsetD8Async` calls initialize outputs 5..9 to `0xff` and counters 10..13 to `0x00`. Exactly four launches follow on that stream:

1. gate `q5_linear`: grid `(16,1,1)`, block `(256,1,1)`, record0/input3/output5/counter10;
2. up `q5_linear`: grid16, block256, record1/input3/output6/counter11;
3. activation `bf16_lut_activation`: grid2, block256, gate5/up6/LUT4/SiLU7/activation8/counter12;
4. down `q5_linear`: grid64, block256, record2/activation8/down9/counter13.

Exactly nine `cuMemcpyDtoHAsync_v2` calls copy outputs 5..13 to their same-index pinned buffers in that order and exact byte counts, followed by exactly one `cuStreamSynchronize`. No default stream, managed memory, memory pool, peer API, mapped-zero-copy kernel access, CUDA Runtime allocation/copy/launch API, CuPy ownership or implicit copy is allowed.

## CUDA identity and context lifecycle

Execution uses a fresh dedicated child process and one owner OS thread. The expected device set has exactly one eligible NVIDIA device: name `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`, PCI `0000:01:00.0`, compute capability 12.0, observed driver 13020; it must be distinct from Intel PCI `0000:00:02.0`. Driver/runtime/compiler version drift is a blocked capability until a new revision; no fallback device is allowed. Start free device memory, queried before any allocation, must be at least 64 MiB.

The context state machine is exactly the GO-audited PH1-R2 contract: `cuInit(0)` and select; `cuCtxGetCurrent(prior)` with `prior=NULL`; diagnostic `cuDevicePrimaryCtxGetState`; exactly one `cuDevicePrimaryCtxRetain`; exactly one `cuCtxPushCurrent_v2` on the owner thread; post-push current-pointer identity; then all stream/module/allocation/copy/launch/sync calls on that owner thread. A non-null prior context blocks before ownership.

Acquisition after push is one nondefault `cuStreamCreate`, one direct `cuModuleLoadDataEx` from frozen cubin bytes, exactly two `cuModuleGetFunction` calls, fourteen pinned allocations, then fourteen device allocations. Cleanup attempts all thirty ordinary releases despite any earlier error: fourteen device frees in reverse order, fourteen pinned frees in reverse order, module unload, stream destroy. It then attempts exactly one pop and requires `popped==owned`, requires restored current context `NULL`, and attempts exactly one `cuDevicePrimaryCtxRelease_v2` last. Reset, context destroy, duplicate retain/release, release-before-pop and any use after release are forbidden. Failure before push releases a retained context once; failure after push attempts every owned release, pop/restore and release. Any cleanup error is negative/invalid, never a scientific negative.

## Precompiled cubin and proof chain

Physical execution cannot compile. A separate future compile-only phase freezes one CUDA C++ source with exactly two `extern "C"` entries, exact options `--std=c++17`, `--fmad=true`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`, `--gpu-architecture=sm_120`, `--device-as-default-execution-space`, direct NVRTC 13.3 DLL/API, complete build log, a diagnostic `compute_120` PTX and an `sm_120` ELF cubin from `nvrtcGetCUBIN`. The compile process may not call `cuInit`, load a module or access a device.

The source, options, NVRTC DLL/version, log, PTX and cubin are create-new and hash-bound before physical authorization. Independent static audit requires the exact width-8 DAG and activation algorithm, no fast-math option, no `-use_fast_math`, no CUDA Runtime/CuPy path, no dynamic parallelism and no third entrypoint. The PTX parser requires exactly the two entrypoints, no `.ftz` instruction modifier, no approximate/transcendental instruction, exact `fma.rn.f32` and `add.rn.f32` reduction operations, and the expected global counter atomics only. Exact hash/version-bound `cuobjdump` and `nvdisasm` raw SASS outputs are retained; parsers require both entrypoints, sm_120 code, no unresolved external call, no device transcendental, and exact source-to-entry/launch correspondence. The cubin ELF magic, byte count and SHA are checked again before `cuModuleLoadDataEx`.

## Controls, resources and terminal evidence

Before loading `nvcuda.dll`, N0 reruns exactly 22 safe controls: for each of the three records, truncation->size, wrong projection->identity, stale CRC->CRC, recomputed-CRC code mutation->canonical digest, recomputed-CRC scale mutation->canonical digest, recomputed-CRC field31->field31, wrong input->input digest; plus one wrong-LUT-digest control. Each retains exact requested/presented metadata and proves `cuda_dll_load=context=module=stream=allocation=launch=0`. The three frozen numerical one-step witnesses are diagnostic only and do not add controls.

Host resource samples, with no telemetry error, occur at exactly: process start, post authorization, post CPU package, post controls, pre CUDA init, post context push, post module/stream, post allocations, post memset/H2D, post four launches queued, post D2H/sync, post thirty ordinary releases, post context release and post serialization. Host available RAM must be at least 16 GiB at start and at least 2 GiB thereafter; retained peak working set must be at most 12 GiB. `cuMemGetInfo` is retained from post-push through post-context-release; requested device bytes are exactly 2,185,216, all live owned resources end at zero, and final free memory may not be more than 64 MiB below preallocation free memory. Evidence artifacts, including raw five outputs, four counters, ledgers, compile provenance hashes and controls, are capped at 16 MiB.

Authorization is invalid without the exact R8V1R1A PASS, R8A5 bundle, CPU package, context-design/audit and PH0X-R12 provenance. Invalid authorization writes nothing. An authorized run is one attempt: a create-new temporary result is verified independently before atomic commit; failures are bounded, create-new and preserve primary error plus complete attempted cleanup. A positive requires every input, control, identity, compiler-provenance, context, buffer, copy, launch, output, counter, resource and cleanup gate true. A device-opened mismatch is a valid negative component result; predevice/protocol/lifecycle/resource failure is infrastructure/invalid, not a scientific negative. No retry, retune or fallback is allowed.

## Claim boundary

A positive N0 supports only: one official real expert-50 Q5 MLP on one known natural activation is reproduced exactly by the pinned NVIDIA direct-cubin path and agrees with the independent CPU oracle and immutable Intel execution. It does not support router/MoE/layer/model correctness, held-out or generalized quality, concurrency, heterogeneous cohabitation, timing, throughput, energy, deployment, novelty or breakthrough claims.
