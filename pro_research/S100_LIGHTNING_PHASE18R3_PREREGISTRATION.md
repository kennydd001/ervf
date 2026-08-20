# S100 Lightning Phase 18R3 preregistration

Date: 2026-08-20.

## Frozen upstream evidence

The authoritative Phase18R2 result is the local
`pro_research/results/s100_lightning18r2/S100_LIGHTNING18R2_SUMMARY.json`.
This phase refuses to start unless:
`flags.SURGERY_RELEASED == true`.

The R2 matched dual CAL64 persistent-cache E mean and reference mean are loaded
from that file at runtime; they are not copied as hard-coded floating-point
constants.

## Protocol

For every arm:
- calibration trace, 10 prompts, 64 targets, 8 warmup;
- production parent graph for prompt;
- role graph for target/probe positions;
- `rt.reset()` only between prompts;
- same process for A/record/O1/OH/B/O2;
- forced O2 so the estimator is exactly alpha .75/.50;
- exact semantic parity on target ids, final logits, hidden/recurrent state,
  KV bytes/state and device position;
- 10,000 prompt-clustered bootstrap resamples inherited from Phase17.

## Reproduction bridge

`FULL_E_ALL` runs first.

Open:
- parity all green;
- abs(R3 E mean - R2 matched E mean) <= 0.75 ms;
- abs(R3 reference mean - R2 matched reference mean) <= 0.75 ms.

No other surgery result is authoritative if the bridge is closed.

## Core arms

Whole-model direct arms:
`ROUTE_ALL`, `SHARED_ALL`, `SCAN_ALL`, `TAIL_ALL`, `ACCUM_ALL`.

Large direct arms:
`UP_ALL`, `DOWN_CORE_ALL`.

Fallback fixed MoE layer groups:
- G0 = [1,3,6,8,10,13]
- G1 = [15,17,20,22,24,27]
- G2 = [29,31,34,36,38,40]
- G3 = [43,45,47,49,51]

Always run `UP_G0..G3` and `DOWN_CORE_G0..G3`.
Group sums are descriptive unless a direct all-layer arm is both parity-green
and reference-stable.

## Stage semantics

ROUTE:
skip exact gate-W GEMV and route_topk; replay ids/w; real cache assignment and
all later computation stay live.

SHARED:
skip both shared-expert GEMVs; replay the exact shared contribution.

UP:
real cache assignment + cache_fetch + resident scale-plane fetch; skip only the
batched routed up GEMV and replay exact post-ReLU2 routed activations.
Therefore it is a direct UP-compute ceiling and a lower bound on the combined
routed-up/fetch opportunity.

SCAN:
skip threshold max/threshold/mask scan. Replay exact masks and rebuild
plist/pcount/nz/nzc deterministically from the recorded masks. The replacement
load+rebuild cost is measured in the overhead arm and added back.

DOWN_CORE:
real route/cache/up/scan and miss-only scale-plane preparation; skip nonzero
column gathers, masked down GEMVs and batched partial reduction; replay exact
per-expert contrib vectors; real weighted accumulation remains.

TAIL:
same live prefix as DOWN_CORE; skip gather/down/reduce and weighted
accumulation; replay exact final MoE output.

ACCUM:
all computation through reduced routed contrib remains real; skip only final
weighted accumulation; replay exact final MoE output.

## Residency gate

For every arm:
`REFERENCE_STABLE = abs(reference_mean_ms - R2_reference_mean_ms) <= 0.75`.

A direct arm that fails this is measured but not quantitative.

## Materiality

`MATERIAL_DIRECT_STAGE_OPEN` iff:
- bridge open;
- direct all-layer arm measured;
- full parity;
- reference stable;
- corrected one-sided 95% lower bound >= 0.50 ms/token.

## Layer atlas

23 independent FULL_E single-layer arms.
A layer is `MATERIAL_LAYER` if lower95 >= 0.20 ms/token.
No sum of single-layer values is treated as a total E estimate.

## Claim boundary

R3 proves exact removable-cost ceilings. It does not prove a production kernel
can realize the oracle saving, and it never sets `S100_SINGLE_ACHIEVED=true`.
