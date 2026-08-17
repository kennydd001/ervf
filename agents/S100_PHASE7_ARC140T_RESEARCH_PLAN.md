# S100 Phase 7 — Arc 140T heterogeneous breakthrough lab

Date frozen: 2026-08-17.

## Starting point

The current ERVF runtime is a CUDA/CuPy runtime on the NVIDIA RTX with
system/host RAM used as a weight tier. The Intel Arc 140T is not currently an
execution device in that runtime.

The phase-4 frozen quality-green baseline is QFAST:

- 18.75165 ms/useful token;
- 53.32864 tok/s;
- 10,240-target V18 fidelity green.

The purpose of this phase is not to add Arc TOPS to RTX TOPS. It is to discover
which *role* the Arc can execute with fewer bytes/boundaries than the current
RTX+RAM design.

## First-principles topology

Three physical/logical tiers:

1. `RTX_VRAM` — small, high-bandwidth GDDR, CUDA/SM120 Tensor Cores.
2. `ARC_UMA` — Intel Arc 140T compute over system DRAM, with XMX/DL-Boost
   available through appropriate Intel software paths.
3. `CPU_RAM` — the same physical DRAM capacity pool as ARC_UMA, but CPU compute.

The Arc does **not** add an independent RAM pool. Its opportunity is compute
locality to bytes that already live in shared DRAM.

## Hypothesis ladder

### H0 — endpoint characterization

Measure CPU, RTX, Arc Vulkan, Arc SYCL when available, and Arc OpenVINO. This
establishes actual target-hardware bandwidth/kernel behavior instead of
importing public numbers.

### H1 — contiguous CUDA+Vulkan layer split

For GGUFs larger than comfortable RTX capacity, sweep RTX:Arc from 5:95 through
95:5, reverse device order, and compare against both endpoint baselines.

### H2 — phase-specific placement

Prefill is GEMM-heavy and can exploit Arc/XMX more effectively; batch-1 decode
is generally bandwidth/latency-sensitive. Test a prefill-friendly hybrid
placement separately from the decode-optimal placement.

### H3 — Arc as cold-expert miss engine [high priority]

Do **not** permanently put every expert on Arc. Keep hot expert tensors/cache on
RTX. For a routed expert that is not resident in RTX VRAM, send the small
hidden/activation vector to shared memory, execute that expert against existing
system-RAM weights on Arc, and return only the hidden-size contribution. This
trades MB-scale PCIe weight migration for KB-scale activation/result traffic.

### H4 — Arc as sparse weight coalescer/prefetch engine

If Arc expert compute itself is too slow, use it as a DRAM-side preprocessing
engine: read scattered compressed panels from UMA, compact exact bytes into
contiguous pinned staging, prefetch when prediction permits, and issue
fewer/larger H2D transfers. Do not expand PCIe bytes with dequantized staging.

### H5 — split attention by context age

For long context, keep a hot/recent KV window on RTX and cold/old KV in
ARC_UMA. Send only Q to Arc. Arc computes old-KV online-softmax partials and
returns per-head statistics/value accumulators for exact combination on RTX.

### H6 — Arc draft engine + RTX target verifier

Run a small draft model/head on Arc while RTX executes the target. Transfer
token ids/probabilities, not weights. Required metric is accepted useful target
tokens per second, never raw draft tok/s.

### H7 — ATSInfer-style tensor scheduling

Tensor scheduling means tensor-granular placement/scheduling, **not** llama.cpp
`--split-mode tensor`. Cost each tensor/operator as compute + transfer + sync +
cache miss + opportunity cost, with different plans allowed for prefill/decode.

### H8 — cross-vendor tensor parallelism [negative-control research]

Use only on a supported dense sanity model. It is not the primary ERVF route.

### H9 — architecture-correct XMX low-bit kernel

Probe OpenVINO/SYCL/Vulkan capability and matched projection shapes first. Only
build a custom Intel low-bit kernel after the exact 140T probe proves a useful
primitive; do not copy adjacent architecture kernels blindly.

### H10 — NPU sidecar [low priority]

Inventory the NPU, but consider it only for coarse predictor/router/draft-side
tasks; per-layer NPU crossings are unlikely to help.

## Automatic Phase-7 test families

The one-click pack inventories hardware/backends, measures CUDA pinned transfers
and RAM bandwidth, bootstraps OpenVINO and matched Arc projection geometries,
refreshes/builds llama.cpp CUDA+Vulkan when possible, discovers GGUFs, measures
CPU/RTX/Arc endpoints, auto-fit plus full forward/reverse layer-ratio sweeps,
VRAM headroom, KV types, context depths, Flash Attention, op-offload,
experimental tensor compatibility, and produces scheduler break-even rankings
for cold experts, coalescing, cold KV, drafting and three-tier scheduling.

If no suitable GGUF exists, hardware/OpenVINO/transfer experiments still run
and GGUF-dependent hypotheses are explicitly marked unmeasured.
