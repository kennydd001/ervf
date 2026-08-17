# S100 phase 8 — Arc 140T routed-downflow engine

Date: 2026-08-17

## Corrections after phase 7

Phase 7's `M=6` OpenVINO result reused one weight matrix across six rows. That is free-M/batch evidence, not six different routed experts. It also used a stale K=3072 proxy. Phase 8 discovers the live ERVF shape contract; the current routed intermediate is 1856 (116 panels × 16), hidden width 2688.

## Primary hypothesis — Arc routed-down engine (ADE)

Keep routing and routed-up on RTX. Move only the six routed down projections to Arc. RTX writes six small ReLU² vectors to shared system memory. Arc reads six different panel-major NVFP4 expert records directly from UMA, applies route weights, and returns one 2688-element sum. This targets the exact stage whose weights are otherwise sparsely gathered over PCIe every token.

## Required experiments

1. live QFAST cache-miss/route census;
2. same-weight M scaling vs N distinct expert weights;
3. export real panel-major NVFP4 records and real activations;
4. custom OpenCL NVFP4 routed-down kernel on Arc 140T;
5. CUDA-pinned host pages wrapped by Intel OpenCL `USE_HOST_PTR`;
6. QFAST latency while Arc is saturated;
7. D3D12 cross-adapter shared-heap/fence capability.

## Promotion

End-to-end integration opens only if the strict real-NVFP4 N=6 kernel passes numerical gates, kernel+bridge is <=0.25 ms/layer, and Arc contention does not erase the expected win. Component benchmarks never count as S100.
