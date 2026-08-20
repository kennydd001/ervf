# S100 Lightning Phase 18 — MoE surgery

Date: 2026-08-20
Parent: Phase-17 exact oracle atlas, Lightning-only.

## Why Phase 18 exists

The complete MoE oracle removes ~10.68 ms/token from a ~19.39 ms reference and by itself crosses the 10 ms S100 target as a ceiling. MoE is therefore the first component for which the measured end-to-end ceiling is large enough to justify structural engineering.

The goal of Phase 18 is not to invent a faster MoE kernel immediately. It is to discover **which exact subpaths and which layers account for the 10.68 ms**, including interaction effects. Only then is a production kernel selected.

## Current production MoE anatomy

For each of 23 MoE layers the quality parent performs, in order:

1. F32 router GEMV;
2. top-k routing and route weights;
3. device-cache assignment;
4. routed-up code/scale fetch on the copy stream;
5. H-SCALE down-scale-plane fetch on the copy stream;
6. shared-expert up projection + ReLU2;
7. shared-expert down projection;
8. routed-up batched NVFP4 projection + ReLU2;
9. thresholded panel scan;
10. six pipelined sparse down-code gathers;
11. six masked routed-down kernels;
12. partial reduction;
13. route-weight accumulation.

This means one MoE layer can contain roughly twenty GPU/copy nodes. Across 23 layers the node count is itself a first-class hypothesis.

## Phase 18A — per-layer exact oracle atlas

Using the already-recorded exact MoE output tables from Phase 17, replay one MoE layer at a time and measure all 23 layers under the frozen ten-prompt workload.

Required outputs per layer:

- corrected saving ms/token;
- one-sided 95% prompt-clustered interval;
- fraction of total E-oracle saving;
- A/B drift and thermal slope;
- replay overhead;
- parity: hidden, recurrent, KV, final logits, token ids.

Then measure cumulative top-N replay sets for N={1,2,4,8,12,23}, where ranking is frozen from calibration/prompt-A only and verified on the full workload.

Purpose:

- identify hot layers versus a uniform ~0.46 ms/layer tax;
- test layer additivity;
- estimate how much of E can be recovered by optimizing only a subset.

## Phase 18B — exact subpath oracles

Record the necessary intermediate tensors in a recorder pass and run the following exact sub-oracles.

### R — router

Replay exact route ids and route weights, but retain cache assignment and the complete expert computation.

Ceiling includes:
- router F32 GEMV;
- gate bias/top-k selection.

### S — shared expert

Replay exact shared-expert output vector while retaining routing and all routed experts.

Ceiling includes:
- shared up NVFP4 GEMV;
- ReLU2;
- shared down NVFP4 GEMV.

### U — routed-up plus up-weight fetch

After real routing/cache assignment, replay the six exact post-ReLU2 routed-up activation vectors and skip routed-up weight fetching/compute. Keep H-SCALE plane fetch, panel scan and routed-down path real.

Ceiling includes:
- routed-up code/scale miss fetch;
- routed-up NVFP4 compute.

### P — threshold/panel scan

Run routed-up normally, then replay exact masks, panel lists, panel counts, sparse column lists/counts and max-activation metadata.

Ceiling isolates the thresholded scan/compaction machinery.

### D — complete routed-down path

Run routing, cache assignment, shared expert and routed-up normally. Replay the exact final route-weighted routed contribution vector before it is merged with the shared output.

Ceiling includes:
- H-SCALE scale-plane fetch;
- sparse down-code gathers;
- six masked down projections;
- partial reduction;
- route-weight accumulation.

### RD — routed expert total

Replay the exact complete routed-expert contribution, bypassing U+P+D while retaining router/cache assignment and shared expert.

### C — expert-cache/control path

Replay route ids/weights but preserve current cache assignment. Measure a control in which equivalent device copies are executed while the original cache/fetch path remains active. This is an overhead ceiling for cache/control rather than a semantic bypass if a clean exact replay is not possible.

## Phase 18C — subpath interaction lattice

Minimum required combinations:

- U+D;
- U+P+D;
- S+D;
- S+U+D;
- R+U+D;
- R+S+U+P+D.

The final combination must agree with the complete E oracle within its replay-overhead confidence interval. If it does not, an unmeasured MoE subpath remains and Phase 18 cannot proceed to kernel design.

Report interaction:

`interaction(X,Y) = saving(X+Y) - saving(X) - saving(Y)`.

Positive interactions are especially important because Phase 17 proved E+A super-additivity.

## Phase 18D — node-count / launch-overhead control

MoE has many small kernels per layer. Build two controls:

1. an empty-node graph containing the same number/order of trivial device kernels/events as the MoE path;
2. a replay-copy graph with one fixed 2688-F32 copy per MoE layer.

Report the difference between graph-node scheduling overhead and actual MoE work. This determines whether the next implementation should prioritize fusion/persistence versus arithmetic/data movement.

## Phase 18E — physics accounting

For every subpath report:

- VRAM bytes read/written/token;
- mapped-host/PCIe bytes/token;
- measured effective GB/s;
- kernel/node count;
- required speedup to hit S100 when combined with measured A-oracle and, separately, A+L oracle ceilings.

Do not use theoretical FLOPS as the primary ranking metric.

## Decision thresholds

Using the Phase-17 parent scale:

- >=2.0 ms lower-confidence saving: structural primary target;
- 0.58–2.0 ms: immediate secondary target;
- 0.20–0.58 ms: only implement if low risk or super-additive;
- <0.20 ms upper-confidence: close independently.

For a production MoE redesign to open, the measured subpath set must show a plausible implementation bundle that recovers at least 60% of the E-oracle saving, or at least 50% when combined with realistic attention/LM-head work that crosses 10 ms.

## Candidate production architectures after the atlas

These are hypotheses, not preselected winners:

### Persistent fused MoE-layer kernel

Fuse router/top-k, shared expert, routed-up, threshold generation, down reduction and route accumulation into as few kernels as dependency/data-placement constraints allow. The explicit goal is to collapse the current ~20-node MoE layer to 2–4 graph nodes.

### Six-expert super-GEMV / warp-specialized routed-up

All six routed experts consume the same normalized hidden vector. Process their row-major NVFP4 up matrices in one persistent kernel, reuse the activation/LUT state and emit threshold metadata directly, eliminating the separate routed-up and panel-scan launches.

### Fused sparse downflow

Consume the six sparse activation streams and H-SCALE planes in one warp-specialized kernel, overlap staging of next expert/panel with current FMA work and produce the route-weighted hidden contribution directly. Eliminate per-expert down kernels, partial buffers and the final route-accumulation launch.

### CUDA-native copy/control path

If cache/fetch/control is large, replace host-synchronized or fragmented staging with graph-captured fixed-buffer transfers and device-side bookkeeping. DirectHost is already a measured negative and must not be resurrected unchanged.

### Native SM120 NVFP4 Tensor Core path

Only opens if the arithmetic oracle shows routed/shared compute, rather than PCIe/fusion overhead, is dominant. Quality must be revalidated because reduction order changes.

## Publication and evidence

The local Phase-17 implementation/results must first be pushed intact to `agent/s100-lightning-phase17-oracle-atlas`.

Phase-18 executable source and compact results go to:

`agent/s100-lightning-phase18-moe-surgery-hardware`.

Large replay tables remain local and are represented by hashes/manifests.

No Phase-18 microbenchmark or oracle result itself claims S100. S100 requires a production-path <=10.000 ms/useful token with frozen quality green.