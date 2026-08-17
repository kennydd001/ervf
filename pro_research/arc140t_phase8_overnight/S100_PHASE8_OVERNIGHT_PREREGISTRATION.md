# Phase 8 overnight preregistration

Default wall budget: 8 hours.

A) Record Intel OpenCL device/driver/work-group and external-memory/semaphore capabilities.

B) Run 8192 real QFAST causal tokens and preserve actual cache need[] misses, route overlap, hot-set coverage and LRU curves.

C) Capture offsets 0,1,4,16, all 23 MoE layers, using runtime-discovered hidden/intermediate/top-k dimensions.

D) On every snapshot test real panel-major NVFP4 on Arc for N={1,2,4,6}, strict/fast-math and local={64,128,256}. Strict gate: finite, cosine>=0.999, NRMSE<=0.02.

E) Rebuild the same offsets independently and measure the current RTX H-SCALE sparse gather -> masked down -> reduce/route-accumulate path. Cross-check all route IDs.

F) Export one complete real down-expert bank, preload it persistently on Arc, rotate random distinct six-expert sets across it and scrub 128 MiB before timed kernels. Use cold/warm ratio as a conservative cache-pressure correction.

G) Repeat the exact-size CUDA-pinned/OpenCL bridge and BASE/real-Arc-load/BASE QFAST contention throughout the night.

H) ADE_GO only when correctness, route identity, pressure-adjusted >=10% speed advantage, <=5% median contention, <=10% first-to-last drift and >=3 independent reruns all pass. ADE_NO_GO on correctness failure, >=10% slower Arc downflow, or >15% sustained QFAST regression. Otherwise ADE_BORDERLINE.

No component projection counts as S100.
