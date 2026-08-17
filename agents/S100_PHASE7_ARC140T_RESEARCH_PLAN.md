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

This is the lowest-boundary hybrid baseline.

### H2 — phase-specific placement

Prefill is GEMM-heavy and can exploit Arc/XMX more effectively; batch-1 decode
is generally bandwidth/latency-sensitive. Test a prefill-friendly hybrid
placement separately from the decode-optimal placement.

### H3 — Arc as cold-expert miss engine [high priority]

Do **not** permanently put every expert on Arc.

Keep hot expert tensors/cache on RTX. For a routed expert that is not resident
in RTX VRAM:

- send the small hidden/activation vector to shared memory;
- execute that expert against the existing system-RAM weights on Arc;
- return only the hidden-size expert contribution;
- execute RTX-resident experts concurrently when dependencies allow.

This trades MB-scale PCIe weight migration for KB-scale activation/result
traffic. It is the most direct way for the 140T to attack the current
host-weight bottleneck.

### H4 — Arc as sparse weight coalescer/prefetch engine

If Arc expert compute itself is too slow, use it as a DRAM-side preprocessing
engine:

- read scattered panels/columns from UMA;
- compact the exact compressed bytes into contiguous pinned staging;
- optionally predict/gather next-layer expert panels;
- issue fewer/larger H2D transfers while RTX computes current work.

No dequantized intermediate is allowed if it expands PCIe bytes.

### H5 — split attention by context age

For long context, keep a hot/recent KV window on RTX and cold/old KV in
ARC_UMA. Send only Q to Arc. Arc computes attention partials for old KV and
returns per-head online-softmax statistics/value accumulators. RTX combines
cold and hot partials exactly.

This is not a short-context speed trick. It targets context capacity and
long-context attention traffic without transferring the old KV cache over PCIe
each token.

### H6 — Arc draft engine + RTX target verifier

Run a small draft model or draft head on Arc while RTX executes the target.
Transfer token ids/probabilities, not weights. This avoids stealing RTX
bandwidth for drafting. It becomes especially interesting if native SM120 FP4
can verify multiple target positions for near-M1 weight-stream cost.

Required metric is accepted useful target tokens per second, never raw draft
tok/s.

### H7 — ATSInfer-style tensor scheduling

“Tensor scheduling” here means tensor-granular placement/scheduling, **not**
llama.cpp `--split-mode tensor`.

Cost every candidate tensor/operator using:

`compute(device) + transfer(bytes) + sync + cache miss + opportunity cost`.

Allow different schedules for prefill and decode. The Phase-7 lab records the
measurements needed to fit this cost model.

### H8 — cross-vendor tensor parallelism [negative-control research]

Current llama.cpp tensor mode does multiple cross-device reductions and is not
implemented for Nemotron-H-MoE/Mamba hybrid architectures. Run it only on a
supported dense sanity model when available. It is a measurement/control, not
the primary ERVF integration route.

### H9 — architecture-correct XMX low-bit kernel

The 140T silicon has Intel GPU AI acceleration, but backend exposure is the
question. Probe OpenVINO/SYCL/Vulkan capability and matched projection shapes.
Only build a custom Xe-LPG+/XMX low-bit kernel after the probe proves a useful
primitive. Do not copy Xe2/140V instructions blindly.

### H10 — NPU sidecar [low priority]

Inventory the NPU, but only consider it for predictors/router/draft-side tasks
with coarse synchronization. Per-layer NPU crossings are unlikely to be useful.

## What Phase 7 actually tests automatically

1. Hardware, RAM, driver, power and device inventory.
2. Current S100 baseline/result discovery.
3. CUDA pinned host<->RTX transfer latency/bandwidth from 4 KiB to 64 MiB.
4. CPU/system-memory copy bandwidth sanity.
5. OpenVINO Arc device and projection-geometry probes when OpenVINO can be
   installed/imported.
6. Current llama.cpp source refresh/build attempt for CUDA+Vulkan.
7. Device enumeration with duplicate RTX-Vulkan rejection.
8. Every discovered GGUF:
   - CPU endpoint;
   - RTX endpoint;
   - Arc Vulkan endpoint;
   - alternative Arc endpoint (SYCL/OpenVINO) when the build exposes one;
   - layer auto-fit;
   - 19 forward RTX:Arc layer ratios;
   - 19 reversed device-order ratios;
   - CUDA launch queue 4x test;
   - best-placement KV f16/q8_0/q4_0;
   - best-placement context depth 4k/16k/32k where the model/runtime permits;
   - Flash Attention auto/on/off;
   - op-offload on/off;
   - tensor mode smoke only as an experimental compatibility test.
9. A scheduler/break-even analysis for expert-compute, coalescer, KV-cold-tier,
   draft-engine and layer-split hypotheses.
10. One compact summary and raw result directory.

If no suitable GGUF exists, the hardware/OpenVINO/transfer experiments still
run and the summary marks GGUF-dependent hypotheses as unmeasured rather than
guessing.
