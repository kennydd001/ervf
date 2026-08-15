# HET-NEXT-L0-PH1 implementation design

Date: 2026-08-13  
State: design-only companion to the PH1 preregistration; no physical action authorized.

## Components

1. `ph1_common.py`: constants, allowlisted range reads, exact record format/checker, pure integer IEEE FP32 FMA/add/BF16 conversion, exact BF16 multiply, LUT builder, official fused-gate-up CPU source graph, separate-record exact-DAG CPU Q5 graph, controls, manifests and atomic transaction helpers. It imports no device runtime.
2. `ph1_intel.py`: ctypes OpenCL backend owned by one thread. It accepts already verified immutable package bytes, exposes no standalone CLI, rechecks a runtime authorization object before its first OpenCL DLL/API access, compiles fixed source, and executes the four-kernel in-order host-USM chain.
3. `ph1_nvidia.py`: direct CUDA-driver cubin backend owned by one thread. It accepts the same verified package, exposes no standalone CLI, rechecks authorization before first CUDA/CuPy access, loads frozen cubins only, and executes the exact pinned/device chain.
4. `run_ph1.py`: sequential state machine `provenance -> CPU/control/source-quality -> Intel -> Intel cleanup/verification -> NVIDIA -> NVIDIA cleanup -> precommit verification -> commit`.
5. `verify_ph1.py`: independent implementation with no imports from components 1-4. It reconstructs source, codec, LUT, bit-level arithmetic, metrics, ledgers, identities, lifecycle and mutation negatives.
6. `preflight_ph1.py`: AST/bytecode/text audit plus pure simulators only. It imports no torch, safetensors, OpenCL, CUDA, CuPy, model or checkpoint module and opens no payload.

Every source, preregistration, design, upstream evidence item, dependency/version lock, cubin, and verifier lock is SHA-bound before outputs open. Closed and opened execution states use distinct immutable locks and acknowledgement tokens.

## Exact buffers

Each device uses this logical table; Intel has one host-USM instance, NVIDIA has one pinned-host plus one device instance.

| name | bytes | producer / consumer |
|---|---:|---|
| gate record | 675,840 | host -> gate linear |
| up record | 675,840 | host -> up linear |
| down record | 675,840 | host -> down linear |
| natural input | 4,096 | host -> gate/up |
| BF16 SiLU LUT | 131,072 | host -> activation |
| gate words | 1,024 | gate -> activation/evidence |
| up words | 1,024 | up -> activation/evidence |
| SiLU words | 1,024 | activation -> evidence |
| activation words | 1,024 | activation -> down/evidence |
| down words | 4,096 | down -> evidence |
| gate counters | 2,048 | gate -> evidence |
| up counters | 2,048 | up -> evidence |
| activation counters | 2,048 | activation -> evidence |
| down counters | 8,192 | down -> evidence |
| **total** | **2,185,216** | exact; no aliases or unlisted allocations |

NVIDIA combined pinned+device allocation is exactly `4,370,432` bytes. The five H2D operations are the three records, input and LUT. The nine D2H operations are gate/up/SiLU/activation/down and four counter arrays. Stage and counter initialization uses exactly nine device memsets. Intel uses no copy API.

## Kernel topology

- Gate/up Q5: grid/global `16 x 256`, logical tile/subgroup width 8, 512 output rows. Each physical subgroup uses a correct width-8 tile primitive; constant `0xff` masks for later warp subgroups are forbidden.
- Activation: grid/global `2 x 256`, one row per thread, 512 outputs. It checks finite gate/up/LUT result, indexes LUT by the raw gate word, applies the integer BF16 multiplication routine, writes SiLU and activation, increments exactly one counter.
- Down Q5: grid/global `64 x 256`, width 8, 2048 rows. It consumes the device activation pointer directly.
- All outputs initialize to `0xffff`; all counters initialize zero.

OpenCL source is compiled with explicit no-fast-relaxed-math and denormal-preservation requirements. CUDA source is compiled to sm_120 cubin through direct NVRTC with no fast math and `--ftz=false`; the cubin/PTX/SASS audit must show no FTZ arithmetic. Cubins are generated and hash-frozen in a compilation-only phase before physical authorization. Direct CUDA driver loading of cubin is permitted; runtime source compilation and textual PTX JIT are forbidden in the physical phase.

## Fail-closed state machine

The CPU phase constructs all evidence and calls the independent verifier in CPU-only mode. A hash-bound `cpu_committed` eligibility artifact is required by both device backends. Intel can open only with the opened token and passing eligibility hash. NVIDIA can open only with the same token plus the independently verified Intel result and clean-release hash. Any malformed prior output, stale temp, failed source/control/quality gate, device error, non-exact tensor, counter/canary error, resource breach, or cleanup error stops the chain and writes immutable failure evidence after cleanup.

The verifier recomputes results from retained bytes and never trusts `positive`, byte totals, metrics, identities, counters or cleanup summaries. The final adjudicator accepts only a fresh independently written PASS artifact whose SHA is embedded in the commit marker.
