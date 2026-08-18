# S100 phase 9 — final findings and next bottleneck

Date: 2026-08-18 · Branch: agent/s100-phase9-repair-hardware ·
Runner: `pro_research/RUN_S100_PHASE9_FULL.ps1` (one-click) /
`pro_research/run_s100_phase9_full.py`.

## Repair outcome

All phase-9 instrumentation is green (`S100_PHASE9_REPAIR_VERIFY.json`: PASS;
summary `instrumentation_complete: True`; full-suite runner exit 0 with
`failures=[]`). Three root causes were found in the real error JSONs and fixed:

1. **Capacity arms**: the fail-fast check `planned_plane_bytes >
   mem_info.free` in `s100_phase9_capacity_runtime.py` was a false negative.
   It compared 492.4 MiB of planned resident scale planes against 459.0 MiB of
   point-in-time free VRAM while the CuPy caching/graph pools still held
   hundreds of MiB that a real allocation reuses. Probe evidence
   (`probe_s100_phase9_build_mem.py`): the same 1656-slot build, run to
   completion, ends with 289 MiB headroom. The pre-check is replaced by the
   real allocation attempt; a genuine OOM is now reported as a clean
   `infeasible_vram` verdict instead of a traceback. In practice no arm hit
   OOM — even the 2034-slot map builds (with WDDM spill to the 64 GB host).
2. **RTX staged-vs-DirectHost probe**: `cp.cuda.alloc_pinned_memory` rounds
   the allocation up (14,271 KiB -> 16 MiB); reshaping the raw buffer to
   `(6, 2494464)` raised. The probe now reshapes only the requested prefix
   (same pinned base pointer, same UVA contract).
3. **Arc miss probes**: never executed and `pyopencl` was absent from every
   venv. Installed in `.venv`; the OpenCL kernel runs on the Arc Pro 140T
   with cosine >= 0.999999 and NRMSE <= 4.8e-6 against the staged RTX
   reference for all five layers, N=1/2/3.

## Measured verdicts (all gates as preregistered)

| Experiment | Result | Verdict |
|---|---|---|
| Capacity budget_neutral A/C/C/B | base 18.175 ms, cand 18.261 ms, all 8 gates green | below gate (cand slower) |
| Capacity plus_128 A/C/C/B | base 19.110 ms, cand 19.585 ms; parity/repeat/finite/vram green; base drift 1.029 ms > 1 ms gate | measurement_failed (marginal drift; cand slower regardless) |
| Capacity plus_256 A/C/C/B | base 18.576 ms, cand 21.048 ms, gates green | below gate |
| Capacity plus_379 A/C/C/B | base 18.274 ms, cand 22.575 ms, gates green | below gate |
| RTX DirectHost vs staged, N=1/2/3 | bit-exact (max_abs 0.0) on all 15 rows, but +18..+61% slower | DIRECTHOST_PROMOTE = False |
| Arc 140T miss engine vs staged | N=1: -9.0% (slower), N=2: +21.8%, N=3: +17.1% | ARC_MISS_PROMOTE = False (N=1 below the +10% gate) |
| Markov prefetch (budgets 4/8/12) | precision ~2% (gate: >=40%), demand misses 10.29% -> 10.14% | PREFETCH_RESEARCH = False |
| Belady oracle at current map | 5.10% vs 10.29% production LRU | headroom exists, see below |

Best measured config remains the current map at ~18.06 ms/token
(55.4 tok/s). S100 single not achieved.

## Next biggest first-principles bottleneck

Phase 9 closes the cache/miss path as a route to 100 tok/s, with numbers:

- Total serial miss-fetch cost at the current 10.29% miss rate is
  **1.508 ms/token** (oracle: `theoretical_current_up_fetch_serial_ms`,
  26.17 GB/s PCIe anchor, 3.118 MB/slot incl. down plane). Even a perfect
  Belady policy (5.10%) caps the win at ~0.75 ms/token ≈ **4% of the
  18.06 ms token** — and the four measured capacity maps show the real,
  overlapped win is smaller still (<=0 ms at the wall).
- The miss-path engines are done: DirectHost is bit-exact but slower; Arc
  wins only at N>=2 and fails the N=1 gate.

The remaining mass is the dense, VRAM-bound side: 2048 MB/token of dense
reads (Mamba 892 + routed-up 387 + shared/gate 290 + attention 281 +
lm_head 198) against a measured 338.4 GB/s streaming roofline give a
**6.05 ms/token floor**, while the dense GEMV suite runs at ~127.9 GB/s
(37.8% of roofline). Actual token time is 18.06 ms. Closing the dense-GEMV
efficiency gap — Mamba's 892 MB/token (43.6% of all VRAM bytes) first — is
worth several times the entire remaining cache-miss budget and is the only
lever whose ceiling reaches the 10.0 ms/token S100 bar (117 tok/s serial,
165 tok/s overlapped, per the two-bus accounting in PATH_TO_100_TOKS.md).

**Next research target: dense GEMV bandwidth efficiency, starting with the
Mamba state scan (892 MB/token, 43.6% of VRAM traffic, never yet
optimized).**
