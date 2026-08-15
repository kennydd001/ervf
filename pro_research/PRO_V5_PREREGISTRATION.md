# PRO V5 preregistration — batch the down_proj sub-kernels across expert slots

Date frozen: 2026-08-16, before any V5 target-hardware measurement.
Base branch state: `59fe46c` (HEAD at freeze time).

Not a reinterpretation of V4. V4 (41.13 tok/s, full mode, all gates pass)
remains the current best verified result and is what V5 is measured against.

## Why this is the next step, and why it's scoped this narrowly

Three prior diagnostics in this session (`diag_component_timing_v4.py`,
`diag_down_subkernels_v4.py`, `RESEARCH_NOTEBOOK.md` 2026-08-16) established,
with direct `cp.cuda.Event` timing rather than static byte-size arithmetic:

- the down-projection pipeline (`down_masked_into_indirect`, called once per
  routed expert per MoE layer) costs 9.57-11.39 ms/token, ~28-30% of the
  eager token time, larger than up-proj's own GEMV (5.00 ms/token);
- split into its four sub-kernels, `gather_down_sparse_ind` (the PCIe
  host-mapped masked read) is the single largest piece at 4.74 ms/token
  (41.6% of the pipeline) — but `panel_scan` + `reduce_partials` together
  cost about the same (4.74 ms/token) with **no PCIe component at all**,
  which points at launch overhead: 4 sub-kernels x 6 expert slots x 23 MoE
  layers = 552 kernel launches per token for down_proj alone, several of
  them (`panel_scan_k` launches with a single `(1,) x (256,)` grid) doing
  very little parallel work per launch.

Verified from `runtime.py` (not inferred): `_moe_dev`'s `for s in
range(top_k):` loop reuses one single-expert-sized `self.mstate` scratch
buffer sequentially across all 6 slots, writing each result into its own
slice of a pre-sized `self.contrib` buffer. There is no cross-slot
dependency — the six expert calls are embarrassingly parallel, only
implemented sequentially. Batching them (six slots' worth of scratch state,
kernels that grid over the slot dimension instead of being launched six
times) would not change any computed value, only launch granularity.

This preregistration covers **only** that batching (fewer, larger launches).
It deliberately does NOT also try to restructure `gather_down_sparse_ind`'s
PCIe access pattern in the same experiment — that is a separate, harder
problem (the masked/sparse gather is inherently data-dependent on ReLU2
activation sparsity) and mixing it in would violate the one-variable rule
and make a fail hard to diagnose.

## What must NOT be attempted here

- Full device-caching of down_proj panels like up_proj's cache. Already
  shown infeasible: `DOWN_PANEL_BYTES` is 2.68 MB/expert (confirmed from
  `CODE_BYTES`/`SCALE_BYTES`, not the ~1 KB first assumed), and the GPU had
  0 MiB free during the V4 run. Full caching at cap 72 x 23 layers would
  need ~4.4 GiB. Not reopened without a capacity-tradeoff study of its own.
- Any change to `panel_scan`/`gather_down_sparse_ind`/`down_masked_ind`/
  `reduce_partials`'s actual arithmetic. Batching changes only how many
  slots one launch processes, never the per-slot math, the reduction order,
  or the masked-column selection logic.

## Arms

All arms load the model once, rebuild cache state between arms
(`enable_cache`), same pattern as V3/V4's BASE_A/candidate/BASE_B:

- **BASE_A**: current per-slot sequential `down_masked_into_indirect`,
  unmodified, called from `_moe_dev` exactly as today (device_cache=True,
  eager — batching is validated in eager mode first; graph capture is a
  later, separate integration step once eager parity holds, mirroring how
  V4 layered selective ERVF onto graph-safe only after V3-G1B proved it
  eager first).
- **BATCHED**: `_moe_dev` monkeypatched (bound-method replacement on the
  runtime instance, the same non-invasive pattern `_install_selective` used
  for V3/V4 — no edit to `runtime.py`/`fused_nvfp4.py`) to call new batched
  kernels that process all `top_k` slots in one launch per sub-kernel type
  per layer, using 6x-sized scratch buffers instead of `self.mstate`.
- **BASE_B**: BASE_A restored, to measure drift.
- **CTL**: a deliberate corruption in the batched path (e.g. swap the
  expert-id-to-slot mapping fed to the gather kernel for two slots) that
  must produce a detectably wrong token sequence. If CTL does not diverge,
  the correctness gates below have no discriminating power and the result
  is void regardless of what BATCHED measured (rule 8).

## Hard correctness gates

1. `batched_equals_base_bitexact` — BASE_A, BATCHED and BASE_B token ids
   identical over a real causal rollout (>=3 prompts, >=64 tokens each,
   reuse the V3/V4 anchor+code prompt set for continuity).
2. `base_drift_le_1ms` — BASE_A vs BASE_B p50 drift <=1.0 ms.
3. `ctl_diverges` — the corrupted CTL arm produces a different token
   sequence than BASE_A on at least one prompt.
4. `launch_count_reduced` — structural check (via `cupy.cuda.profiler` range
   markers, or a direct count of kernel-launch Python calls per token in
   eager mode) confirming the batched path issues fewer down_proj-related
   launches per token than BASE_A's 552.
5. `extra_vram_lt_64MiB` — batching's 6x scratch buffers must not blow the
   VRAM budget (they are small: masks/plist/nz are `O(intermediate)` per
   slot, six of them is still well under a MiB).

## Speed gate

Full mode (>=500 timed decode samples): BATCHED p50 improves over BASE_A p50
by >=1.0 ms or >=3%. Smoke-mode speed is diagnostic only, matching V3/V4
convention.

## Kill criterion — a valid technical closure, not a problem

If the batched kernels cannot be made bit-exact against the sequential
per-slot reference after reasonable debugging (i.e. the masked-gather /
reduction logic does not transfer cleanly to a slot-batched grid), that is
recorded as a closure exactly like G2's `cudaGraphLaunch`-is-not-capturable
finding: a real result, not a gap to paper over. No gate is widened to make
a partial win pass.

## Claim boundary

Even a full pass here does not, by itself, establish a path to 100 tok/s.
Rough arithmetic in `RESEARCH_NOTEBOOK.md` (2026-08-16) puts the ceiling of
eliminating the entire down_proj pipeline at roughly 65-75 tok/s from V4's
24.3 ms/token baseline — this experiment can only close part of that gap
(the launch-overhead half, ~4.74 ms/token of the 11.39 ms/token pipeline),
since it explicitly does not touch the PCIe-bound `gather_down_sparse_ind`
term. Do not restate a BATCHED pass as a tok/s claim beyond a fresh
integrated causal A/B, per the project's standing rule against upgrading
component measurements to token-level claims.
