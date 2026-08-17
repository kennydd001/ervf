# S100 phase 8 — Arc 140T routed-downflow engine

Date: 2026-08-17

## Correction to phase 7

Phase 7 measured `expert_down_M6` as six rows multiplied by **one shared
weight matrix**. That is a useful free-M / batch-amortization result, but one
single-stream Nemotron token routes to six **different** experts. It therefore
cannot be promoted as evidence that top-6 routed down computation costs 0.162 ms.

Phase 7 also used K=3072 for the expert-down geometry. The live ERVF record
format proves the routed intermediate is 1856 on this checkpoint:

- down panels: 116 × 16 = 1856;
- hidden width: 2688;
- one NVFP4 matrix: 2,494,464 code bytes + 311,808 scale bytes.

Phase 8 discovers the dimensions from the live runtime and never hard-codes a
stale model family geometry.

## Breakthrough hypothesis A — Arc routed-down engine

Current RTX path:

1. RTX computes six routed up projections and ReLU²;
2. each routed down expert lives in host/shared RAM;
3. sparse weight columns/scales are gathered over PCIe to RTX;
4. RTX computes six masked down projections;
5. route-weighted contributions are reduced.

Proposed ADE path:

1. RTX keeps routing and routed up projections;
2. six ReLU² vectors are written to a tiny shared buffer;
3. Arc reads the six **different** NVFP4 down records directly from UMA;
4. one Arc kernel computes all six sparse down projections, applies route
   weights and emits one 2688-element sum;
5. RTX imports the sum and continues.

The payload crossing is tens of KiB, not MiB of weights.

## Breakthrough hypothesis B — cache-miss full-expert engine

The live route/cache census measures `need[]` from the existing device cache.
If misses are sparse enough, an alternate scheduler can keep cache hits on RTX
and execute only misses on Arc, overlapping Arc work with hot RTX experts.

## Breakthrough hypothesis C — common pinned host bridge

CUDA pinned host pages are system memory. Intel OpenCL can potentially wrap the
same pointer with `CL_MEM_USE_HOST_PTR`. If accepted by the driver, a single
page-locked buffer can be:

RTX D2H -> Arc kernel -> RTX H2D

without a CPU memcpy. This is measured directly, not assumed from UMA.

## Breakthrough hypothesis D — D3D12 cross-adapter shared buffer

Windows D3D12 supports cross-adapter shared heaps/fences in system memory.
Phase 8 contains a capability probe. A green result opens a later CUDA
external-memory + OpenVINO/Intel-GPU RemoteTensor implementation, potentially
removing even the explicit host-copy abstraction from the scheduler.

## Required evidence

- live hidden/intermediate/expert counts;
- actual cache miss distribution over real causal tokens;
- same-weight M scaling versus N distinct expert matrices;
- actual current-model panel-major NVFP4 records and ReLU² activations;
- Arc custom NVFP4 sparse-down timing on those records;
- numerical comparison against an independent CPU decoder;
- CUDA-pinned/OpenCL bridge timing;
- RTX QFAST interference while Arc is saturated;
- optional D3D12 cross-adapter heap/fence capability.

No component benchmark is an S100 result. End-to-end integration follows only
when the measured economics are positive.
