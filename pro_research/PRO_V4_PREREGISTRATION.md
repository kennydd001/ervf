# PRO V4 preregistration — integrated graph-safe + selective-ERVF arm

Date frozen: 2026-08-16, before any V4 target-hardware measurement.
Base branch state: `852ab3b PRO V3 smoke results` (HEAD at freeze time).

This is a new experiment namespace. It does not reinterpret or weaken the V3-G0S
or V3-G1B results. Both remain valid on their own terms: G0S measured graph
residency alone (+10.1%, 28.6063 ms), G1B measured selective ERVF alone
(+10.73%, 28.158 ms). Neither may be added arithmetically to the other — that is
exactly the open question V4 answers physically.

## Why V4 is the correct next step

`graph_safe_v3.py`'s `GRAPH_SAFE` arm captures one token body
(`_step_body_graph`) into a CUDA graph. That body calls `self._attention` and
`self._mamba`, which dispatch every dense GEMV through `self.k.mv_bf16` /
`self.k.mv_fp8_tensor` / `self.k.mv_f32` (`runtime.py` lines 401-474). CUDA
graph capture records whatever kernel launch happens at capture time through
those Python call sites — it does not care whether the attribute currently
points at the production kernel or a monkey-patched one.

`selective_ervf_v3.py`'s `_install_selective(rt, dense)` replaces exactly those
three attributes on `rt.k`, dispatching the four frozen winning shapes to
`DenseERVF` and leaving every other shape on the production kernel.

Therefore: calling `_install_selective(rt, dense)` **before** `rt.setup_graph()`
should capture the ERVF kernels for Q/O (BF16) and Mamba in/out (FP8-tensor)
directly into the replayed graph, while K/V/router keep the production kernels
inside the same graph. This is not a new mechanism — it is the two already-
measured mechanisms sharing one capture, which is the only way to learn their
physical interaction (Amdahl overlap, launch-overhead sharing) instead of
guessing from two separate p50s.

## Arms

- **EGR**: production kernels, device-cache eager, no graph. Identical
  construction to G0S/G1B (`enable_cache` → `load_routed_bank` →
  `device_cache=True` → `deterministic_accum=True`).
- **GRAPH_SELECTIVE**: `_install_selective(rt, dense)` runs first, then
  `rt.setup_graph()` captures the token body with ERVF kernels bound at the
  four frozen shapes. Prompt tokens are staged through `step_graph(id)` with a
  stream sync after each one (the V3-G0S repair), exactly as GRAPH_SAFE does.
  Decode timing excludes prompt sync, unchanged from G0S.
- **DET**: two GRAPH_SELECTIVE rollouts from `reset()`, must be identical.
- **CTL**: recapture with `rt._bad_pick = 1` (the existing G0 sabotage hook)
  baked into the same selective-dispatch graph. Must diverge from the EGR
  reference on at least one prompt within the first 64 tokens.

Selective dispatch is restored (`restore()`) after CTL capture so no other
process state leaks.

## Frozen shapes (unchanged from V3-G1B, not re-tuned here)

- BF16 ERVF: `(4096, 2688)`, `(2688, 4096)` — attention Q, O.
- FP8-tensor ERVF: `(10304, 2688)`, `(2688, 4096)` — Mamba in_proj, out_proj.
- FP32 router and every other BF16/FP8 shape: production kernel only.

## Hard correctness gates

1. `argmax_direct_tie` — the existing two-pass argmax kernel selects the
   expected low-index winner in the synthetic tie test (reused from G0S).
2. `graph_dot_contains_ervf` — `rt._graph.debug_dot_str()` contains both
   `pro_gemv_bf16_ervf16` and `pro_gemv_fp8_tensor_ervf16`. This is the
   structural proof that ERVF kernels were actually captured into the graph,
   not merely exercised during the pre-capture warmup call.
3. `graph_selective_equals_egr` — GRAPH_SELECTIVE ids == EGR ids for every
   generated token, every prompt.
4. `graph_selective_deterministic` — DET A == DET B.
5. `bad_pick_control_diverges` — CTL differs from the EGR reference on at
   least one prompt within the compared window. A pass on this gate that
   silently succeeds without this diverging is not trusted (rule 8).
6. `extra_vram_lt_64MiB`.

No threshold is learned from the V4 run; the four ERVF shapes and the 2.5 ms /
64 MiB bars are carried over unchanged from V3/G0.

## Speed gates

Smoke (3 prompts × 16 tokens, 45 timed samples): diagnostic only, matching the
existing convention that 16-token rollouts cannot adjudicate a p50 claim.

Full (3 prompts × 256 tokens, >=500 timed samples):

- `full_speed_gain_ge_2_5ms`: GRAPH_SELECTIVE p50 <= EGR p50 - 2.5 ms (the
  same absolute bar G0/G0S used for graph residency alone).
- `full_samples_ge_500`.

Reported as **informative, not gating** (comparing across separate sessions
with different thermal/cache history is not a controlled A/B): GRAPH_SELECTIVE
p50 against the previously recorded G0S GRAPH_SAFE p50 (28.6063 ms smoke) and
G1B SELECTIVE p50 (28.158 ms smoke). If GRAPH_SELECTIVE full-mode p50 is at or
below the lower of the two, that is evidence the mechanisms compose without
negative interaction; if it lands above both, that is evidence of overlap/
contention and must be reported as such, not hidden.

## What this does not claim

- Not a 50 tok/s claim. That still requires a long rollout, tails, VRAM and
  thermal behavior per the pack's breakthrough threshold in
  `PRO_HYPOTHESES.md`.
- Not a claim that the two mechanisms' gains are additive. The whole point of
  physically integrating them is to measure the true combined number instead
  of assuming `2.8931 + 3.3841` ms.
- Does not touch K/V/router kernels, gatherless downflow, or any closed idea
  already dead per `agents/STATE_OF_THE_WORK.md`.

## Model-identity note (carried from the V3 anchor investigation)

All V4 arms run under `pro_research`'s default model directory
(`nemotron_3_5_lightning_v35`, the officially adjudicated
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` checkpoint — see
`agents/RESEARCH_NOTEBOOK.md`, 2026-08-16 entry, for the full identity
investigation). External comparison against `V36_DETERMINISTIC_ANCHOR.json` is
informative-only and is expected to diverge from token 1, because that anchor
was frozen against `models/nemotron_3_5_lightning` — confirmed to be
`NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` mislabeled on disk, a different
checkpoint. A GRAPH_SELECTIVE-vs-EGR mismatch is a real bug; a
GRAPH_SELECTIVE/EGR-vs-V36-anchor mismatch, on its own, is not.
