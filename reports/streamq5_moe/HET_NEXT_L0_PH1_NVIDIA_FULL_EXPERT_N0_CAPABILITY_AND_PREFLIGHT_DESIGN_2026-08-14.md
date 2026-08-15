# PH1 NVIDIA full-expert N0 capability and preflight design

Date: 2026-08-14. This document defines future gates only. It authorizes no import, CPU payload read, compiler, CUDA load or device call.

## Phase separation

N0 is split into five immutable namespaces. No phase may promote or mutate artifacts from another.

1. **S0 source freeze:** runner, direct-Driver backend, CPU-only independent verifier, failure/transaction helper, CUDA source template and static preflight sources are frozen. All execution/compile locks are closed.
2. **P0 static preflight:** parses sources as text/AST; validates hashes, ABI declarations, exact constants, source mutation tests, transaction/failure TEMP simulations and output absence. It may not import the runner/backend/CUDA source module, read the shard/D2 payload, load NVRTC/nvcuda/cudart, enumerate a device or compile.
3. **C0 compile-only:** after independent S0/P0 audits, a fresh process uses direct NVRTC only to create diagnostic PTX and execution cubin. It may not load nvcuda/cudart, call `cuInit`, create a context, load a module or enumerate/access a device. Independent compile verification freezes all compiler artifacts and SASS-tool outputs.
4. **C1 capability-only:** after compile audit, a fresh child may load the Driver API and run a tiny unrelated integer sentinel under the exact primary-context lifecycle, solely to establish identity, ABI, context, cubin load and cleanup capability. It cannot open the shard, D2, CPU package or real cubin. Capability evidence cannot satisfy scientific gates.
5. **E0 physical correctness:** only after separate final authorization, one fresh child reconstructs the exact CPU package, runs controls, then performs the one real NVIDIA execution. The independent verifier starts only after the child exits and uses CPU/read-only evidence. No physical retry follows any terminal artifact.

If capability-only execution is judged unnecessary, it may be omitted; its absence cannot weaken E0 gates. Compile and physical namespaces remain distinct either way.

## P0 static requirements

The preflight lock binds the N0 preregistration/design, R8V1R1A verification SHA `42cd6958...`, exact R8A5 bundle hashes, CPU commit/manifest/stages/LUT, PH1-R2 context contract/audit and PH0X-R12 cubin lessons. The preflight independently rejects any source containing model loading/forward, Transformers, safetensors tensor materialization outside the three frozen byte ranges, CuPy, CUDA Runtime allocation/copy/launch, default stream, primary reset, context destroy, managed memory, pool, peer, graph, event-timing or performance code.

AST/call-surface checks require exact ctypes bindings and return types for every used Driver function: `cuInit`, device count/get/name/PCI/attribute/driver version, current-context get, primary-state/retain/release, push/pop, stream create/destroy/synchronize, module load/get-function/unload, host alloc/free, device alloc/free, async H2D/D2H, async D8 memset, launch, and memory-info. `_v2` exports are used where the installed API exposes them; every `CUdeviceptr` is uint64 and every kernel argument is a pointer to a stable host variable. No raw device address is truncated or passed as the parameter-array slot itself.

Static source tests independently reconstruct the exact 14-buffer table, 5 H2D/9 memset/4 launch/9 D2H schedule, byte totals, stable argument maps, launch geometry, width-8 row mapping, BF16 rounder, exact integer BF16 multiply and all normative multiply vectors. Targeted mutations change each size/direction/shape/loop distance/field order/rounding tie/entry name/option and must fail.

The actual transaction and failure helpers are redirected to TEMP and exercised for: clean create/verify/commit; existing valid commit; stale temp; corrupt/missing/extra file; interrupted result/manifest/commit; failure before and after device-open; cleanup secondary error preserving primary; oversized partial; quarantine collision; and repeated invocation. Invalid authorization must leave TEMP empty. All globals are restored in `finally`.

## C0 compiler contract

Direct NVRTC is loaded cdecl from one exact DLL path with every called function `argtypes/restype` frozen. Options and source bytes are exact. Program creation, compile, log retrieval, PTX size/data, cubin size/data and destroy return codes are retained. Program destroy is attempted even after compile/retrieval failure. A nonempty ELF cubin, nonempty diagnostic PTX and complete log are hard gates; output files use create-new atomic transactions. Source, compiler DLL, version, options, log, PTX, cubin, `cuobjdump`, `nvdisasm` and both raw disassemblies receive SHA-256 identities.

The independent compiler verifier reparses source/options, PTX and SASS without importing the physical backend. It validates the two entrypoints and instruction restrictions from the preregistration, exact ELF architecture and no executable third entry. Mutations remove/change entrypoints, inject `.ftz`, fast-math, approximate transcendental, runtime API, wrong architecture, wrong DAG distance, wrong code/scale offset and wrong launch geometry; every mutation must fail. C0 output does not authorize a device run.

## C1/E0 capability and identity gates

On Windows the process must observe exactly one eligible device: `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`, PCI `0000:01:00.0`, compute capability `(12,0)`, driver version `13020`; Intel evidence remains PCI `0000:00:02.0`. The implementation records UUID and total/free memory but makes no UUID claim until a separately frozen capability result. Any count/name/PCI/capability/driver drift blocks; no ordinal fallback.

The owner native thread ID and every Driver ledger row retain QPC timestamps and current-context pointer. Context rows are exactly: prior-current query, primary-state query, retain, push, post-push current query, pop, restored-current query, primary release. Ordinary resource rows remain separate. A verifier state machine rejects non-null prior, pointer mismatch, duplicate/missing/out-of-order row, use from another thread, release-before-pop, reset/destroy, use-after-release and any nonzero cleanup code. Failure injection is required at every acquisition and release position; each already-owned resource is released at most once and every remaining release is attempted.

Capability-only sentinel, if used, is 1,024 uint32 words with a frozen bijection and raw output, not a Q5 kernel. It uses a separately compiled cubin/hash and fresh buffers. No C1 artifact can be substituted for real N0 input or output.

## E0 evidence schema and independent verification

The physical result schema has exact nonempty sections: invocation/process; dependency hashes; CPU preparation identities; 22 controls; CUDA identity; compiler/cubin provenance; context ledger; ordinary resource ledger; buffer/copy/launch ledger; raw five BF16 stage arrays; raw four uint32 counter arrays; stage/counter manifests; resource samples; cleanup; gates; terminal classification. Every raw array records dtype, shape, bytes and SHA. All retained scalars must be finite and type-exact.

Expected cardinalities are hardcoded independently: 14 pinned allocations, 14 device allocations, 9 memsets, 5 H2D, 4 launches, 9 D2H, 1 sync, 30 ordinary release attempts, 8 context rows, 14 resource samples and 22 controls. The verifier reconstructs the records from the official source ranges, reruns the exact CPU oracle, rechecks every output/counter byte, pointer/argument identity, order, device identity, current-context state and cleanup. It does not import the candidate runner/backend/codec.

The verifier has mutations for each top-level field, raw byte/hash, gate, cardinality, pointer alias, H2D/D2H direction/bytes, launch entry/geometry/stream, context pointer/order, release code/order, live resource, control checker stage, resource bound and terminal class. Empty lists/maps are never valid. Exact positive, allowed device mismatch, predevice infrastructure, protocol, lifecycle and cleanup cases are mutually exclusive. Canonical positive output is create-new and exists only if all checks pass.

## Resource and artifact policy

Hard host bounds are 16 GiB available at process start, 2 GiB minimum thereafter and 12 GiB maximum retained peak working set. Device start-free is at least 64 MiB; requested device bytes are exactly 2,185,216. All buffers and context/module/stream ownership must be zero after cleanup. The final-free diagnostic is at least preallocation-free minus 64 MiB. Result plus raw evidence is at most 16 MiB; failure evidence is separately capped at 16 MiB and records cleanup/dispositions. No artifact may be overwritten.

No latency, throughput, clock, power or performance measurement appears in any N0 phase. Timing fields exist only for lifecycle order and bounded timeout diagnostics.
