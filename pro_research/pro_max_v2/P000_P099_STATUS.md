# P000–P099 post-V6 disposition

Frozen against `pro-research` commit `5c699300da2d10552f5037426c1607119b2239b4`.

This is a pruning table, not a claim that every DONE item is a publication-grade result. `DONE` means the repository contains a physical result covering the original question; `PARTIAL` means useful evidence exists but the original scope is not fully closed.

| ID | Original hypothesis | Post-V6 status | Evidence / next action |
|---|---|---|---|
| P000 | Identity/provenance lock | **DONE** | Correct Lightning/Nano identity and source hashes are documented. |
| P001 | Reproduce V4 graph+selective baseline | **SUPERSEDED** | V4 reproduced and surpassed by verified V6 at 21.0923 ms/token. |
| P002 | CUDA graph node census | **DONE** | V4/V6 graph DOT probes enumerate captured mechanism names. |
| P003 | Recompute 50/75/100 tok/s budgets | **DONE** | PATH_TO_100_TOKS recomputes single-stream and aggregate ceilings. |
| P004 | Fuse six route-order accumulations | **DONE** | V6 uses exact route-order batched accumulation. |
| P005 | Fuse Q/K/V projections into one exact ERVF launch | **OPEN-E50** | Post-V6 mixed Q/K/V one-launch candidate; implemented in this pack. |
| P006 | Fuse residual add with next RMSNorm | **OPEN-E50** | Post-V6 add + next RMSNorm candidate; implemented in this pack. |
| P007 | Integrate all individually passing exact fusions | **OPEN-E50** | Physical combination of passing post-V6 exact candidates. |
| P008 | Nsight Systems/Compute availability and graph profile | **PARTIAL** | Many diagnostics exist; no single canonical post-V6 Nsight census. |
| P009 | Clock, power and thermal baseline | **PARTIAL** | Clock collapse was measured; one-hour thermal validation remains open. |
| P010 | Six-slot batched down projection | **DONE** | V5/V6 batch six-slot down path. |
| P011 | Batch panel scan across six routes | **DONE** | Panel scan batching is in V6. |
| P012 | Persistent coalesced sparse gather | **OPEN-ARCH** | Batched gather was tested; persistent producer/consumer gather remains distinct and open. |
| P013 | Gather vector-width autotune | **DONE** | Scalar versus wide mapped-host access was measured; uchar4 path adopted. |
| P014 | Gather work-grid autotune from live nzc/pcount | **OPEN-E50** | Live nzc/pcount work-grid autotuning not closed. |
| P015 | Batch masked down partial GEMV across slots | **NEGATIVE** | Batched gather/down-masked variants were exact but not worth VRAM/latency in V6. |
| P016 | Fuse partial reduction with route-order accumulation | **DONE** | Exact partial reduction/route-order accumulation batching in V6. |
| P017 | Overlap gather(s+1) with GEMV(s) | **OPEN-ARCH** | Gather(s+1) versus GEMV(s) overlap not integrated. |
| P018 | Double-buffer sparse down mirrors | **OPEN-ARCH** | Double-buffer sparse down mirrors not integrated. |
| P019 | Producer-consumer host-to-SMEM down kernel | **OPEN-ARCH** | No-bounce host-to-SMEM producer/consumer fabric remains a main architectural route. |
| P020 | TMA/native toolchain capability | **OPEN-PROBE** | TMA/native capability must be measured on this Windows/CUDA stack. |
| P021 | Mapped-host/TMA direct-to-shared microbenchmark | **OPEN-PROBE** | Mapped-host direct-to-shared/TMA microbenchmark not performed. |
| P022 | Cluster multicast of one host tile to row CTAs | **OPEN-MOONSHOT** | Cluster multicast not tested. |
| P023 | Static hot down-expert cache under fixed VRAM | **PARTIAL** | Full down-cache is infeasible; a small static hot-panel cache remains open. |
| P024 | Split fixed VRAM budget between up and down caches | **OPEN-ARCH** | Joint up/down cache budget not physically optimised. |
| P025 | Layer-heterogeneous up-cache oracle | **DONE** | Per-layer capacity oracle/sweep exists. |
| P026 | Physical heterogeneous cache A/B | **DONE** | Budget-neutral physical capacity A/B completed. |
| P027 | Route-stationarity across prompts/domains | **DONE** | Prompt/domain and warm-cache route behaviour measured. |
| P028 | Per-layer hot-expert coverage | **DONE** | Per-layer locality/hot coverage measured. |
| P029 | One-token-ahead exact expert prefetch | **NEGATIVE** | One-token route prediction missed its preregistered recall gate. |
| P030 | Latency-aware dynamic cache budget controller | **OPEN-ARCH** | Dynamic latency-aware cache controller not built. |
| P031 | Concurrent Q/K/V streams inside graph | **OPEN-E50** | Q/K/V concurrency remains open; one-launch QKV is tested first. |
| P032 | Overlap Mamba conv1d and dt activation | **OPEN-E50** | Mamba conv and dt branch overlap not integrated. |
| P033 | Fuse Mamba elementwise stages | **OPEN-ARCH** | Mamba elementwise fusion not built. |
| P034 | Persistent Mamba SSM kernel across layers | **OPEN-MOONSHOT** | Persistent Mamba kernel not built. |
| P035 | Fuse gated norm with Mamba output projection | **OPEN-ARCH** | Gated norm plus Mamba output projection fusion not built. |
| P036 | Mixed-shape exact QKV ERVF | **OPEN-E50** | Same physical candidate as P005; implemented in this pack. |
| P037 | Fuse attention O projection with residual write | **OPEN-E50** | Attention O projection plus residual write not built. |
| P038 | Fuse KV append with attention read path | **OPEN-ARCH** | KV append plus attention read fusion not built. |
| P039 | Context-dependent attention split autotune | **OPEN-E50** | Context-dependent split autotune remains open. |
| P040 | Page-aware long-context attention queue | **OPEN-LONGCTX** | Page-aware long-context queue remains open. |
| P041 | Hardware FP8 decode audit | **PARTIAL** | FP8 paths were profiled and tuned, but a canonical hardware-decode audit remains open. |
| P042 | Packed-f32 attention association change | **CLOSED** | Packed-f32 association change violates exact arithmetic requirements. |
| P043 | Conditional CUDA graph nodes for dynamic paths | **OPEN-PROBE** | Conditional graph support exists in CUDA but is untested in this runtime. |
| P044 | Low-level cudaGraphAddChildGraphNode epoch graph | **OPEN-E50** | Old nested-capture attempt failed; low-level child graph API is implemented in this pack. |
| P045 | Device-side graph launch | **OPEN-ARCH** | Device-side graph launch not tested. |
| P046 | cudaGraphExecUpdate instead of recapture | **OPEN-ARCH** | GraphExecUpdate path not tested. |
| P047 | Persistent device token loop | **OPEN-BREAKTHROUGH** | Persistent device token loop remains open. |
| P048 | Remove captured event/fence overhead | **OPEN-E50** | Captured event/fence overhead audit remains open. |
| P049 | Stream dependency and unnecessary-wait audit | **PARTIAL** | Launch overhead is proven; exact unnecessary-wait attribution remains open. |
| P050 | Fuse LM-head ERVF with block argmax | **OPEN-E50** | LM-head ERVF plus exact block argmax is implemented in this pack. |
| P051 | Hierarchical exact LM-head top-1 reduction | **OPEN-E50** | Hierarchical exact top-1 is part of P050 implementation. |
| P052 | Certified vocabulary tile skipping | **OPEN-MOONSHOT** | Certified vocabulary tile skipping not built. |
| P053 | Mapped-host embedding gather audit | **DONE** | Mapped-host embedding path is graph-safe and causally verified. |
| P054 | Small hot embedding row cache | **OPEN-E50** | Small hot embedding row cache not tested. |
| P055 | Final RMSNorm/LM-head cooperative fusion | **OPEN-MOONSHOT** | Final norm plus LM-head cooperative fusion remains open. |
| P056 | Residual add + next norm exact fusion | **OPEN-E50** | Same candidate as P006; implemented in this pack. |
| P057 | Norm-weight constant/read-only cache audit | **OPEN-E50** | Norm-weight cache audit not performed. |
| P058 | Elementwise layer-transition megakernel | **OPEN-ARCH** | Layer-transition megakernel not built. |
| P059 | Stable allocator/pointer graph audit | **DONE** | Stable captured pointers/caches are exercised by V4-V6. |
| P060 | Full NVFP4 entropy census | **OPEN-BYTES** | Full current-checkpoint entropy census remains open. |
| P061 | Scale-conditioned random-access ANS | **OPEN-BYTES** | Scale-conditioned random-access ANS remains open. |
| P062 | Fused lossless decode into GEMV registers/SMEM | **OPEN-MOONSHOT** | Fused lossless decode into compute remains open. |
| P063 | Exact MoE symmetry canonicalisation before coding | **OPEN-BYTES** | Exact symmetry canonicalisation remains open. |
| P064 | PathQ active-bytes mixed precision | **OPEN-LOSSY** | PathQ requires separate quality-gated track. |
| P065 | Full-depth mixed-precision quality validation | **OPEN-LOSSY** | Full-depth mixed-precision validation not done. |
| P066 | CertiPlane exact residual pages for fc2 | **OPEN-MOONSHOT** | CertiPlane residual pages remain open. |
| P067 | Exact router top-k margin certificate | **OPEN-MOONSHOT** | Router top-k certificate remains open. |
| P068 | Lossless FP8 scale delta coding | **OPEN-BYTES** | FP8 scale delta coding remains open. |
| P069 | Per-expert random-access compressed pages | **OPEN-BYTES** | Random-access compressed expert pages remain open. |
| P070 | Stock llama.cpp exact hardware baseline | **OPEN-EXTERNAL** | Stock llama.cpp exact-machine baseline not committed. |
| P071 | Differential llama.cpp kernel audit | **OPEN-EXTERNAL** | Differential llama.cpp audit not done. |
| P072 | Port ERVF to llama.cpp fork | **OPEN-EXTERNAL** | ERVF not ported to llama.cpp. |
| P073 | Port graph-safe device routing to llama.cpp | **OPEN-EXTERNAL** | Graph-safe routing not ported to llama.cpp. |
| P074 | Port winning batched-down path to llama.cpp | **OPEN-EXTERNAL** | V6 MoE batching not ported to llama.cpp. |
| P075 | FlashInfer/vLLM kernel parity | **OPEN-EXTERNAL** | FlashInfer/vLLM parity not measured on this machine/model. |
| P076 | TensorRT-LLM/NIM baseline where model identity matches | **OPEN-EXTERNAL** | TensorRT-LLM/NIM identity-matched baseline not measured. |
| P077 | iGPU-resident cold experts | **OPEN-HETERO** | iGPU cold-expert placement not tested. |
| P078 | Intel Arc host-USM low-bit GEMV microbenchmark | **OPEN-HETERO** | Intel Arc host-USM low-bit GEMV not benchmarked. |
| P079 | Exact split reduction tree over iGPU+dGPU | **OPEN-MOONSHOT** | Exact split reduction across iGPU+dGPU not built. |
| P080 | NPU asynchronous draft/control workload | **OPEN-HETERO** | NPU asynchronous workload not tested. |
| P081 | CPU cold-expert execution threshold | **OPEN-HETERO** | CPU cold-expert crossover not measured. |
| P082 | Unified mapped-host expert fabric | **OPEN-BREAKTHROUGH** | Unified mapped-host expert fabric remains open. |
| P083 | DirectStorage/GPUDirect Storage cold-start path | **OPEN-EXTERNAL** | DirectStorage/GDS cold-start path not tested. |
| P084 | Lossless decompression overlapped with transfer | **OPEN-BYTES** | Lossless decompression/transfer overlap remains open. |
| P085 | Current linear MTP speculation | **CLOSED** | Current linear MTP path was physically negative. |
| P086 | Route-union-aware proposal tree | **OPEN-LOWPRIORITY** | Route-union-aware tree requires a new mechanism, not current MTP. |
| P087 | Optimised target tree verifier after exact stack | **OPEN-LOWPRIORITY** | Optimised target verifier remains conditional. |
| P088 | Exact K-token child-graph epochs | **OPEN-E50** | Low-level exact child-graph epochs are implemented in this pack. |
| P089 | Multi-token block decode with one weight sweep | **OPEN-BREAKTHROUGH** | One weight sweep for multiple exact tokens remains open. |
| P090 | Integrated exact >=50 tok/s run | **NEAR** | Current verified record is 47.4107 tok/s; 1.0923 ms remains to E50. |
| P091 | Integrated exact >=75 tok/s run | **OPEN-MILESTONE** | No exact 75 tok/s run. |
| P092 | Integrated exact >=100 tok/s run | **OPEN-MILESTONE** | No exact 100 tok/s run; aggregate batch>1 is the main plausible route. |
| P093 | Long-context >=50 tok/s run | **OPEN-LONGCTX** | No long-context E50 run. |
| P094 | 10,000-token causal validation | **OPEN-VALIDATION** | No 10,000-token post-V6 validation. |
| P095 | One-hour thermal validation | **OPEN-VALIDATION** | No one-hour post-V6 thermal validation. |
| P096 | Tokens/joule and sustained clock report | **PARTIAL** | Clocks measured; sustained tokens/joule report remains open. |
| P097 | Independent raw-artifact verifier | **PARTIAL** | Many local verifiers exist; this pack adds a post-V6 independent verifier. |
| P098 | Second GPU/runtime replication | **PARTIAL** | ERVF replicated across Qwen/Nemotron, not yet a second GPU/runtime full-stack replication. |
| P099 | Novelty audit, paper and upstream patch set | **OPEN-PUBLICATION** | Novelty audit/upstream patch set awaits a stable winning architecture. |
