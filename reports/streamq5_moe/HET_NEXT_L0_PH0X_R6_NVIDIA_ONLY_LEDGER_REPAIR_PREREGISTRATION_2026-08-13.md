# HET-NEXT L0 PH0X-R6 — NVIDIA-only ledger repair preregistration

Date: 2026-08-13. Exploratory completion arm only; no formal PH0 pass.

R6 binds and retains the exact PH0X-R5 prereg/runner, R3 Intel result, R4 CUDA compile/staging diagnostic, official record/input, CPU oracle, CUDA source, buffer sizes, call order, launch geometry and claim boundary. There is no arithmetic, kernel, input, threshold or device-selection change.

R6 solely repairs two independent R5 audit blockers:

1. A positive result requires an exact ordered 24-row ledger: compile; stream create; four named allocations; two memsets; two H2D; one grid16/block256 kernel; two D2H; one synchronize; four reverse device frees; four reverse pinned frees; stream destroy; final cleanup row. Sizes and directions are exact, all pointers are nonzero and unique within their address class, all release codes are zero, and no extra/missing rows are accepted.
2. Any exception is wrapped only after `finally` has attempted every reachable release. The partial identity, exact partial ledger, output/counter bytes obtained so far, module-disposal statement and cleanup errors are retained in the outer immutable failure JSON.

The CuPy RawModule/function are explicitly not part of the manually owned release ledger: CuPy owns their driver-module lifecycle. R6 records `cupy_raii_after_reference_drop` and drops both references followed by `gc.collect()` before stream destruction. No claim of an independently observed module unload is made. Manually owned pinned allocations, device allocations and stream are fully gated.

One clean NVIDIA-only attempt in a new output directory; no Intel API, retry or retuning. Positive and combined claim boundary remain exactly R5: one real projection/input reproduced bitwise, not a full expert/layer/model/performance/concurrency/deployment/novelty/breakthrough claim.
