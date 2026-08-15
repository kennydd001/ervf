# HET-NEXT-L0 PH1 NVIDIA full-expert N1 preregistration

Date: 2026-08-14. State: immutable design-only revision; implementation, static preflight, compiler and device execution are closed.

## Scope and supersession

N1 supersedes only the four design defects identified in `HET_NEXT_L0_PH1_NVIDIA_FULL_EXPERT_N0_INDEPENDENT_DESIGN_AUDIT_2026-08-14.md`, SHA-256 `7af526febaadedc065c9b9bed1ca16d5790a823bfb2e77a447db9880c10f228e`. N0 preregistration SHA-256 `9d86bab43d1a2c16f90cdf5d9d67f073f5dd887c2ba44baaae0452b03f43175c` and N0 capability/preflight design SHA-256 `b01fc7818f74a45b00924f21b0b991a529bebd7ba5f95dafdb96769d75736efc` remain immutable and are inherited except where this revision explicitly replaces text.

The four repairs are exactly: (1) the last device-memory query moves before context pop/release; (2) the compiler protocol is one `sm_120` NVRTC program producing both PTX and cubin; (3) Driver loader, ABI and material call operands are frozen; and (4) the PH1-R1 physical contract and its audit are directly bound. There is no change to scientific inputs, Q5/BF16 arithmetic, stage hashes, buffer sizes, copy/launch order, counters, controls, primary-context ownership, thresholds, resource limits, terminal classes or claim.

## Direct provenance chain

N1 directly binds all of the following:

- PH1-R1 CPU evidence and physical contract `HET_NEXT_L0_PH1_R1_CPU_EVIDENCE_AND_PHYSICAL_CONTRACT_2026-08-14.md`, SHA-256 `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981`;
- independent PH1-R1 physical-contract audit `HET_NEXT_L0_PH1_R1_PHYSICAL_CONTRACT_INDEPENDENT_DESIGN_AUDIT_2026-08-14.md`, SHA-256 `cb295f83e5a49ebdccce9982342af4442fa4a1fbb3607d547a5a6804eaa97cfe`;
- PH1-R2 NVIDIA context contract SHA-256 `dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c` and independent audit SHA-256 `db5c6055bbffec2ea1e38e50f3b1c4d5dbece2deaf74c7ce5110056d05c8f1fe`; R2 supersedes only R1's context-lifecycle ambiguity;
- Intel final verifier SHA-256 `42cd69582a47b8b5f8f4b7f24a696f1d3fcc6fbd49c05d0f61354a57cefc052d` and R8A5 result/manifest/commit SHA-256 values `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50`, `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b`, and `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965`;
- CPU package commit/manifest/stages/LUT SHA-256 values `f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06`, `63f6c842f377fb18738d6016b133c7529803581d0cd661739c0ffd648a82ac54`, `c2fbc4d6c3c400ecb0ac7af36b36c88a1c8122d3066cb123430f934bd750d6a8`, and `a3cbc779f1f1e8b0957c651e6b90a64d506568764ab34f7419ba5cc1ede9daed`;
- PH0X-R12 preregistration, verification, report and result SHA-256 values `0a68cc7edf9acefba70f8f3a067ebcd207693a332ee23bc2d1d147b612d0906d`, `7ee3161f03b79be6475fec9ddc4936a28019632640e81b02ed18ed3b132e1b9b`, `6904cbc34eb935e55e455c8f006e75d0dbc6b1c46247a5c3fbb445a4001e344a`, and `159d77b8fc6d1cac3d2123c09b7f256837c9691a397fa9d052309752d26955bc`.

Every later source, static-preflight, compile and physical authorization lock must bind the R1 pair and R2 pair directly, not merely through this document.

## Frozen one-program compiler protocol

One future compile-only process loads one exact direct NVRTC 13.3 DLL with the C calling convention. The one-program choice is supported by the installed `nvrtc.h`, SHA-256 `316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21`, whose `nvrtcGetPTX*` and `nvrtcGetCUBIN*` contracts both retrieve output from the previous compilation of the same program and specify that cubin size is zero only for a virtual rather than actual architecture. This matches the official NVRTC 13.3 API documentation at `https://docs.nvidia.com/cuda/nvrtc/`. It creates exactly one program from one frozen CUDA source containing exactly `q5_linear` and `bf16_lut_activation`, compiles exactly once with this ordered option vector:

`--std=c++17`, `--fmad=true`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`, `--gpu-architecture=sm_120`, `--device-as-default-execution-space`.

For a successful compile, the exact NVRTC call cardinality and order is: `nvrtcVersion` once; `nvrtcCreateProgram` once; `nvrtcCompileProgram` once; `nvrtcGetProgramLogSize` once; `nvrtcGetProgramLog` once; `nvrtcGetPTXSize` once; `nvrtcGetPTX` once; `nvrtcGetCUBINSize` once; `nvrtcGetCUBIN` once; and `nvrtcDestroyProgram` once. Both outputs come from that same program and compile. The textual artifact is named **PTX retrieved from the `sm_120`-targeted compilation**. It is never described as a `compute_120` program or artifact. The cubin must be nonempty ELF for `sm_120`; PTX must be nonempty; the full log is retained even when empty.

If program creation returns a non-null program together with a failing status, that handle is owned immediately and destroy is attempted once. A compile failure still retrieves the available log size/log and destroys the program once; PTX and cubin retrieval are `not_attempted`. After successful compile, any log/PTX/cubin retrieval failure is terminal; later retrieval calls are `not_attempted`, while destroy remains attempted once. Every call records attempted state, return code and output size. A second program, virtual-architecture compile, second compile, runtime JIT, textual-PTX physical load or artifact substitution is forbidden.

The compile verifier hardcodes these success/failure cardinalities, hashes source/options/DLL/version/log/PTX/cubin, parses the PTX and disassembles only the frozen cubin with hash-bound tools. The PTX and SASS gates remain exactly those in N0 and PH0X-R12: no `.ftz`, fast math, approximate/transcendental operation or unresolved external call; exact width-8 FMA/add DAG; two and only two entry points.

## Frozen Driver loader, ABI and call operands

The physical process may load only `C:\Windows\System32\nvcuda.dll`, 4,466,920 bytes, SHA-256 `86b41599a673f1aa4699ab458dc5c1e02b57da64d17221f45327af0393fd59a5`, file/product version `32.0.15.9658`, using `ctypes.WinDLL(absolute_path, use_last_error=True, winmode=0x00000800)` where `0x00000800` is `LOAD_LIBRARY_SEARCH_SYSTEM32`. The file is rehashed before load. The actual loaded-module path is resolved and must equal that path after Windows case-insensitive normalization, and its bytes must rehash identically. No search-name load is allowed.

Every Driver symbol has explicit `argtypes` and `restype=CUresult` (`ctypes.c_int`) under Win64. `CUdevice=ctypes.c_int`, opaque handles are `ctypes.c_void_p`, `CUdeviceptr=ctypes.c_uint64`, `size_t=ctypes.c_size_t`, and flags/counts/dimensions are `ctypes.c_uint`. `CUuuid` is a `ctypes.Structure` containing exactly `ctypes.c_char * 16`. Pointer outputs use pointers to those exact types; every kernel parameter is a pointer to a stable host variable, and the parameter vector is a stable `ctypes.c_void_p * N` array. The exact export/argument table is:

| export | exact `argtypes` |
|---|---|
| `cuInit` | `[c_uint]` |
| `cuDriverGetVersion` | `[POINTER(c_int)]` |
| `cuDeviceGetCount` | `[POINTER(c_int)]` |
| `cuDeviceGet` | `[POINTER(c_int), c_int]` |
| `cuDeviceGetName` | `[c_char_p, c_int, c_int]` |
| `cuDeviceGetUuid_v2` | `[POINTER(CUuuid), c_int]` |
| `cuDeviceGetPCIBusId` | `[c_char_p, c_int, c_int]` |
| `cuDeviceGetAttribute` | `[POINTER(c_int), c_int, c_int]` |
| `cuDeviceTotalMem_v2` | `[POINTER(c_size_t), c_int]` |
| `cuMemGetInfo_v2` | `[POINTER(c_size_t), POINTER(c_size_t)]` |
| `cuCtxGetCurrent` | `[POINTER(c_void_p)]` |
| `cuDevicePrimaryCtxGetState` | `[c_int, POINTER(c_uint), POINTER(c_int)]` |
| `cuDevicePrimaryCtxRetain` | `[POINTER(c_void_p), c_int]` |
| `cuCtxPushCurrent_v2` | `[c_void_p]` |
| `cuCtxPopCurrent_v2` | `[POINTER(c_void_p)]` |
| `cuDevicePrimaryCtxRelease_v2` | `[c_int]` |
| `cuStreamCreate` | `[POINTER(c_void_p), c_uint]` |
| `cuStreamSynchronize` | `[c_void_p]` |
| `cuStreamDestroy_v2` | `[c_void_p]` |
| `cuModuleLoadDataEx` | `[POINTER(c_void_p), c_void_p, c_uint, POINTER(c_int), POINTER(c_void_p)]` |
| `cuModuleGetFunction` | `[POINTER(c_void_p), c_void_p, c_char_p]` |
| `cuModuleUnload` | `[c_void_p]` |
| `cuMemHostAlloc` | `[POINTER(c_void_p), c_size_t, c_uint]` |
| `cuMemFreeHost` | `[c_void_p]` |
| `cuMemAlloc_v2` | `[POINTER(c_uint64), c_size_t]` |
| `cuMemFree_v2` | `[c_uint64]` |
| `cuMemcpyHtoDAsync_v2` | `[c_uint64, c_void_p, c_size_t, c_void_p]` |
| `cuMemcpyDtoHAsync_v2` | `[c_void_p, c_uint64, c_size_t, c_void_p]` |
| `cuMemsetD8Async` | `[c_uint64, c_ubyte, c_size_t, c_void_p]` |
| `cuLaunchKernel` | `[c_void_p, c_uint,c_uint,c_uint, c_uint,c_uint,c_uint, c_uint, c_void_p, POINTER(c_void_p), POINTER(c_void_p)]` |

No other Driver export is callable. `_v2` exports are mandatory where named in the table; aliases or Runtime API substitutes reject.

Material operands are frozen:

- every `cuMemHostAlloc` uses flags `0`; mapped/device-pointer host access is forbidden;
- the single nondefault stream is `cuStreamCreate(..., CU_STREAM_NON_BLOCKING)` with flag value `1`; every copy, memset and launch names that stream;
- `cuModuleLoadDataEx(module, cubin_bytes, numOptions=0, options=NULL, optionValues=NULL)` loads only the frozen cubin;
- each `cuLaunchKernel` uses the N0 grid/block, `sharedMemBytes=0`, the owned nondefault stream, its stable `kernelParams`, and `extra=NULL`;
- no event, graph, managed, pool, peer, mapped-zero-copy or default-stream path exists.

CUDA Runtime is not loaded or queried. `cudart_loaded=false` is a hard evidenced gate, imports/calls/loaded-module scans reject CUDART, and `runtime_version` is exactly the string `not_applicable_driver_api_only`. Only NVIDIA Driver, NVRTC and disassembly-tool identities/versions are normative.

## Corrected resource sampling

The fourteen host sample labels remain fixed, with label 12 clarified:

1. `process_start`; 2. `post_authorization`; 3. `post_cpu_package`; 4. `post_controls`; 5. `pre_cuda_init`; 6. `post_context_push`; 7. `post_module_stream_preallocation`; 8. `post_allocations`; 9. `post_memset_h2d`; 10. `post_launches_queued`; 11. `post_d2h_sync`; 12. `post_ordinary_releases_pre_pop`; 13. `post_context_release`; 14. `post_serialization`.

Device-memory fields are `not_attempted` with null free/total/return-code at samples 1--5. Exactly one `cuMemGetInfo_v2` query occurs at each sample 6--12 while the retained primary context is current. Sample 7 is the normative `preallocation_free_bytes`; sample 12 is the normative `final_pre_pop_free_bytes`, after all 30 ordinary release attempts but before pop. The diagnostic gate is `final_pre_pop_free_bytes >= preallocation_free_bytes - 67,108,864`, and requested device allocations remain exactly 2,185,216 bytes. Start-free at sample 6 is at least 67,108,864 bytes.

After sample 12 the owner pops the exact retained context, proves restored current context null and releases the primary context last. Samples 13 and 14 are host-only: `device_query_state="not_attempted"`, `device_free_bytes=null`, `device_total_bytes=null`, `cuMemGetInfo_return="not_attempted"`, and `driver_context_calls_after_primary_release=0`. No Driver context/memory query occurs after primary release. Host RAM/peak and artifact limits are unchanged.

## Unchanged scientific and terminal contract

The exact 14 buffers (2,185,216 bytes per pinned/device class), 9 memsets, 5 H2D, 4 launches, 9 D2H, one sync, 30 ordinary releases, eight context rows, 22 predevice controls, five expected BF16 stage hashes and four all-one counter arrays are inherited byte for byte from N0/R1. The same official expert-50 records, D2R3 `p0_n16` input, LUT, CPU oracle, quality result and no-retune rule apply.

A positive may claim only exact reproduction of one official real expert-50 Q5 MLP input by the pinned NVIDIA direct-cubin path, agreeing with the independent CPU oracle and immutable Intel result. It cannot claim router/MoE/layer/model correctness, held-out/generalized quality, concurrency, heterogeneous cohabitation, timing, throughput, deployment, novelty or breakthrough. No implementation or execution is authorized until this N1 design is independently audited GO.
