# HET-NEXT-L0-PH0-R1 — implementation design

Date: 2026-08-13  
Status: **design only; no runner, import, preflight, compiler or device action authorized**

Normative preregistration is `HET_NEXT_L0_PH0R1_SINGLE_REAL_PROJECTION_PREREGISTRATION_2026-08-13.md`. The old PH0 implementation design (SHA `362c785213aa24109aa2ebf5de7a002d2b93fa5eb7618f13ac504571c7ce5494`) is immutable provenance and superseded.

## Minimal future source split

If independently authorized, new standalone paths will contain: a record builder; a helper-free software-FP32 CPU oracle; an OpenCL host-USM backend; a CUDA backend; a four-phase sequential runner; an independent verifier importing none of them; locks; and a static no-payload/no-device preflight.

State is create-new: `EMPTY→CPU_COMMITTED→INTEL_COMMITTED→INTEL_CLEAN→NVIDIA_COMMITTED→NVIDIA_CLEAN→VERIFIED`. NVIDIA cannot initialize before durable Intel cleanup. A valid commit returns `already_complete`; an error cleans owned resources before atomic failure evidence and never retries.

## Data and memory

CPU reads exactly 2,097,152 official source bytes and 4,096 D2 input bytes after header/range/hash gates. It constructs one 675,840-byte record in anonymous RAM, verifies the frozen pristine hashes/CRC, and produces 512 CPU BF16 words through an independent explicit width-8 DAG.

Intel owns host-USM record/input/output/counters sized `675840/4096/1024/2048` (683,008 total). NVIDIA owns pinned and device buffers with the same four sizes (683,008 each; 1,366,016 combined). Uint32 counters `[512]` are diagnostic and zero-initialized explicitly; they are not a 512-byte bitmap. Outputs and counters are initialized to sentinels and retained raw.

## Exact kernels

Both kernels use 16×256 and compute `row=block*32+thread/8`, `lane=thread%8`. OpenCL requires subgroup size 8. CUDA uses `cooperative_groups::tiled_partition<8>`; each tile does shuffles 4,2,1. Static source audit rejects any warp-wide unbounded shuffle and any field31 test based on a five-byte `0xff` pattern. The unpacker validates every 5-bit field.

The exact arithmetic is pack ownership `lane+8*v`, increasing eight-column FMA, virtual reduction distances 16/8/4/2/1, tile reduction 4/2/1, and one final BF16 RNE. Compiler sources/options/logs/binaries are raw evidence. Fast math and reassociation options are forbidden.

## Exact backend call contracts

Intel uses only host-USM pointers and `clSetKernelArgMemPointerINTEL`; all relevant OpenCL and extension ABI signatures/returns are checked. No `cl_mem` or explicit copy/map/migrate symbol may be called. `clGetMemAllocInfoINTEL` validates all four allocations. Release-all continues through individual errors.

NVIDIA uses one nondefault stream and exactly: two async memsets (output `0xff`, counter zero), H2D record, H2D input, one kernel, D2H output, D2H counters, one synchronize. Ledger entries bind pointer identity, offset, bytes, direction, stream, sequence, return code and completion. Cleanup covers events, four device buffers, four pinned buffers, module, stream and owned runtime/context in an audited release-all loop.

## Checker and controls

A single safe checker implements the frozen order `size→schema/identity→CRC→canonical payload/input digests→field range→dispatch` and increments a call ledger. The eight exact controls in the preregistration invoke that real checker. The sensitivity control alone uses a separate unsafe CPU decoder only after safe rejection and proves `0xbe53→0xbe52`; unsafe code is unreachable from device methods. The independent verifier reconstructs all mutations byte-for-byte and does not trust summaries.

## Device identification

The capability stage enumerates without weights, requires exactly one matching Intel OpenCL device at `0000:00:02.0` and one CUDA device 0 at `0000:01:00.0`, verifies frozen name/vendor/device/subsystem/revision and driver policy, and records the complete raw inventory. Ambiguity exits blocked before payload read/device allocation.

## Independent verifier

After both devices close, the verifier independently rehashes all sources/evidence, rereads only the two allowed payload ranges, requantizes all 1,048,576 fields, reconstructs the exact record and CPU software oracle, recomputes all controls, raw manifests and 512-word/counter comparisons, and validates nonvacuous allocation/call/cleanup/resource/transaction ledgers. It exits nonzero on any false conjunct and cannot import builder/runner/backend/codec helpers.

## Static preflight before any implementation execution

The eventual static preflight must be source-only and must not import OpenCL/CUDA/compiler/model libraries or read payload. It binds every source/lock/doc SHA; AST-checks forbidden model/router/shared/merge/performance/bank paths; derives all sizes including counter-inclusive `683008` and `1366016`; tests all q values, ties, zero groups, pack order, CRC, field31 and each frozen mutation; tests software FMA/add fixtures and exact 16×256 row coverage; audits cooperative-groups tile width 8 and OpenCL subgroup 8; validates exact call cardinalities; fault-injects every cleanup position and transaction transition; mutates every verifier conjunction; and proves execution-closed locks plus absent output paths.

Passing source audit or preflight is not physical authorization.
