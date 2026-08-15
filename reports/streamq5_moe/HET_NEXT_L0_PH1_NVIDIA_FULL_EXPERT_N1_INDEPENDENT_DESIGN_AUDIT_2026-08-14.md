# PH1 NVIDIA full-expert N1 — independent design audit

Date: 2026-08-14  
Scope: read-only audit of the frozen N1 preregistration, capability/preflight design, and design lock against the four N0 NO-GO findings. No implementation, import, payload read, preflight, compiler, CUDA library load, or device call was performed.

## Verdict

**GO for source implementation only.** N1 closes all four N0 blockers without changing the approved scientific input, Q5/BF16 arithmetic, stage hashes, buffer/copy/launch schedule, counters, controls, lifecycle ownership, resource limits, terminal classes, or claim.

This does not authorize static preflight, NVRTC compilation, capability execution, or physical execution. Implemented sources must be frozen and independently audited before a separately authorized no-device static preflight.

## Frozen N1 package

- preregistration: `953e3dba4158c8dafd78f8a072a8df48f924e12457509c88a3247b62faa1eb05`, 12,143 bytes;
- capability/preflight design: `2f9d129a2f299b15057b8da16191a6087215594863db9593a4643c661a20a90a`, 5,280 bytes;
- design lock: `ddbceeb31637060465464b6709f0aa530a1da7871c5ad6288a97d805efc41312`, 5,170 bytes.

All 23 lock bindings match the handed-off frozen files. The lock explicitly closes implementation, static preflight, compiler, capability, and device phases. Before this audit was written, the N1 family contained exactly the three design files.

## Closure of N0 blockers

### 1. Device-memory query is now context-valid

The 14 host sample labels are exact. Device memory is not queried at samples 1–5. Exactly one `cuMemGetInfo_v2` call occurs at samples 6–12 while the retained primary context is current. Sample 7 is the preallocation baseline; sample 12 is after all 30 ordinary release attempts but before context pop.

After sample 12, the owner performs pop, restored-null verification, and primary release. Samples 13–14 are host-only and carry exact null/`not_attempted` device fields plus `driver_context_calls_after_primary_release=0`. This removes the prior invalid post-release query and agrees with the inherited R2 no-use-after-release contract.

### 2. NVRTC protocol is one exact `sm_120` program

N1 freezes one source, one program identity, one compile and one destroy. The positive ledger has exactly ten calls in order: version, create, compile, log-size, log, PTX-size, PTX, cubin-size, cubin, destroy. PTX and cubin must carry the same program identity and arise from the single `sm_120` compile. The textual artifact is no longer described as a `compute_120` compile.

The design also freezes creation/compile/retrieval/destroy failure semantics, including immediate ownership of a non-null handle, log retrieval after compile failure, no later retrieval after an earlier failure, and one destroy attempt. The installed `nvrtc.h` is directly hash-bound. Static/compiler verifiers must reject a second program/compile/destroy or another architecture option.

### 3. Driver loader, ABI and material operands are exact

N1 freezes the absolute System32 NVCUDA path, byte size, SHA, file/product version, `WinDLL` convention, secure `LOAD_LIBRARY_SEARCH_SYSTEM32` flag, pre-load and loaded-module path/hash checks, and a complete typed export table.

Material operands are now normative:

- `cuMemHostAlloc` flags `0`;
- one nondefault `CU_STREAM_NON_BLOCKING=1` stream;
- `cuModuleLoadDataEx` option count `0` with both option pointers null;
- `cuLaunchKernel` shared memory `0`, owned stream, stable parameter-pointer array, and `extra=NULL`;
- exact uint64 `CUdeviceptr` and stable host variables;
- no mapped, managed, pool, peer, event, graph, default-stream, CUDART, Runtime, or CuPy path.

The former nonexistent Runtime-version gate is replaced by `cudart_loaded=false` and `runtime_version="not_applicable_driver_api_only"`. Driver, NVRTC and disassembly-tool identities remain normative.

### 4. PH1-R1 and R2 are directly bound

The N1 lock directly binds:

- PH1-R1 physical contract `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981`;
- PH1-R1 independent audit `cb295f83e5a49ebdccce9982342af4442fa4a1fbb3607d547a5a6804eaa97cfe`;
- PH1-R2 context repair `dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c`;
- PH1-R2 independent audit `db5c6055bbffec2ea1e38e50f3b1c4d5dbece2deaf74c7ce5110056d05c8f1fe`.

Future source/preflight/compile/physical locks are required to preserve all four direct links. R2 remains a narrow lifecycle supersession rather than an implicit replacement of R1 arithmetic and physical terms.

## Unchanged scientific contract

N1 directly retains the terminal Intel PASS and exact R8A5 bundle, CPU package, three expert-50 records, D2R3 `p0_n16` input, normative LUT, five stage hashes, and no-retune quality boundary.

The inherited physical cardinalities remain coherent and exact:

- 14 pinned and 14 device buffers, 2,185,216 bytes per class;
- 9 device memsets, 5 H2D, 4 launches, 9 D2H, and 1 sync;
- exact gate/up/activation/down grid/block geometry and stable pointer maps;
- five exact BF16 output arrays and four all-one uint32 counter arrays;
- 22 predevice safe controls;
- 30 ordinary releases plus the separate eight-row primary-context lifecycle;
- zero live resources, no cleanup errors, bounded host/device resources and artifacts.

The one-program PTX/SASS proof remains no-FTZ, no fast math, no approximate/transcendental path, exact width-8 FMA/add DAG, exact integer BF16 activation, and exactly two entrypoints. Physical execution consumes only the frozen cubin.

## Implementation requirements

The implementation freeze must expose these contracts as data, not comments alone. Its independent source audit and later static preflight must prove:

1. every ABI and operand constant is used on the production path;
2. the cubin bytes live in a stable host buffer through `cuModuleLoadDataEx`;
3. every kernel parameter slot points to a stable typed host variable rather than containing a raw device address directly;
4. failure injection covers every acquisition/release and every compiler retrieval boundary;
5. the actual independent verifier imports no candidate runner/backend and rejects all frozen mutations;
6. every source/lock directly binds N0/N1 audits and the complete R1/R2/Intel/CPU/PH0X chain.

## Claim boundary

N1 is a design GO, not a NVIDIA result. A later positive may claim only exact reproduction of one official real expert-50/input by the pinned NVIDIA direct-cubin path, agreeing with the independent CPU and immutable Intel evidence. It cannot claim performance, concurrency, heterogeneous cohabitation, router/layer/model correctness, held-out quality, deployment readiness, novelty or breakthrough.

