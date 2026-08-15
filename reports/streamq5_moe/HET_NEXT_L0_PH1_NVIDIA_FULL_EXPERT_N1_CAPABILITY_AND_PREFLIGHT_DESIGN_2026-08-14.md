# PH1 NVIDIA full-expert N1 capability and preflight design

Date: 2026-08-14. Design-only; all implementation, preflight, compiler and device authorization is closed.

This N1 document inherits N0 capability/preflight design SHA-256 `b01fc7818f74a45b00924f21b0b991a529bebd7ba5f95dafdb96769d75736efc` and changes only the four items frozen in the N1 preregistration. The N0 independent design audit SHA-256 is `7af526febaadedc065c9b9bed1ca16d5790a823bfb2e77a447db9880c10f228e`.

## Static source and provenance gates

Every future lock and independent verifier must directly rehash the PH1-R1 contract `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981`, R1 audit `cb295f83e5a49ebdccce9982342af4442fa4a1fbb3607d547a5a6804eaa97cfe`, PH1-R2 context contract `dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c`, R2 audit `db5c6055bbffec2ea1e38e50f3b1c4d5dbece2deaf74c7ce5110056d05c8f1fe`, N0 documents/audit, N1 documents and downstream source/lock files. An indirect or missing link fails.

The future no-device static preflight reads sources as text/AST only. It may not import a runner/backend/source module, load the CPU payload, load NVRTC/NVCUDA, compile, enumerate CUDA or open a device. It independently checks the absolute NVCUDA path/hash constants and a complete signature table. Source mutations change every `argtypes`, `restype`, handle type, `CUdeviceptr` width, loader convention/path, `_v2` export, host/stream flag, module option count/pointer, launch shared-memory/extra/stream/parameter pointer and must reject. It rejects `CDLL` for NVCUDA, relative/search-name loads, any CUDART/CuPy/Runtime symbol and any version gate named CUDA Runtime.

The exact static schedule remains 14 pinned plus 14 device allocations, 9 memsets, 5 H2D, 4 launches, 9 D2H and 1 sync. It proves flags `cuMemHostAlloc=0`, `CU_STREAM_NON_BLOCKING=1`, module options `(0,NULL,NULL)`, and launch fields `(sharedMemBytes=0, extra=NULL, owned stream)`. It also proves device-memory calls occur only in sample slots 6--12, that pop/release precede host-only samples 13--14, and that no Driver call can follow primary release.

## One-program compile-only gate

The compile-only process has exactly one NVRTC program lifetime. A positive ledger has the exact ten calls and order from the N1 preregistration: version, create, compile, log-size, log, PTX-size, PTX, cubin-size, cubin and destroy, each once. The one ordered option vector contains `sm_120`; neither source nor evidence may contain a second architecture option or call the PTX a `compute_120` compile.

The compiler result retains source, option-vector, NVRTC DLL/version, installed NVRTC header SHA-256 `316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21`, return codes, complete log bytes, PTX bytes and cubin bytes. It records `program_identity` on every program-bearing row and proves the PTX and cubin came from that same identity. Exact output files are create-new and atomically committed only after an independent compiler verifier passes. Any empty PTX, empty/non-ELF cubin, wrong architecture, second create/compile/destroy, missing destroy, changed option or artifact mismatch fails. Failure-path simulations cover null/non-null create failure, compile failure with log retrieval, each retrieval failure and destroy failure; every row has explicit `attempted`, code and bytes.

The independent parser applies the unchanged no-FTZ/DAG/two-entry gates to PTX from the `sm_120` compile and to SASS disassembled from that cubin. `cuobjdump`/`nvdisasm` tool binaries, versions and raw outputs are hash-bound. No textual PTX is a physical module input; the physical module consumes cubin bytes only.

## Physical resource and lifecycle verification

The exact primary-context state machine remains the R2 eight-row contract. The verifier requires one current-context `cuMemGetInfo_v2` at each resource sample 6--12 and zero at all other samples. It independently recomputes the ordered resource labels and the free-memory diagnostic from sample 7 to sample 12. It requires samples 13 and 14 to carry exact `not_attempted`/null device fields and `driver_context_calls_after_primary_release=0`.

The physical call ledger independently requires absolute `WinDLL` loading of the frozen System32 NVCUDA bytes, actual loaded-path equality, the exact ABI, host flags 0, nonblocking stream flag 1, module options zero/null/null and launch shared-memory 0/extra null. It scans loaded modules and result provenance for `cudart_loaded=false` and `runtime_version="not_applicable_driver_api_only"`. Any Runtime/CUDART load, call or version value fails rather than becoming a device negative.

All N0 verifier requirements remain nonvacuous: exact raw five BF16 arrays and four uint32 counters; independently rebuilt CPU records/oracle; exact pointer/copy/launch identities and cardinalities; 22 controls; resources; 30 ordinary release attempts; eight context rows; zero live ownership; and create-new bounded terminal artifacts. Mutations cover each newly frozen sample/ABI/compiler field in addition to all N0 mutations.

No implementation, preflight, compile, capability or physical call is authorized by these documents. A future source package requires an independent N1 design GO first.
