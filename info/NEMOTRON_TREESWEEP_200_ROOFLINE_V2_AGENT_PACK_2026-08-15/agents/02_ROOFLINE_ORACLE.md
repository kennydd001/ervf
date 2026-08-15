# Agent 02 — Target Tree Roofline Oracle

## Mission

Measure the exact baseline target-only tree cost before any drafter work. This baseline is diagnostic after N1–N5; only Agent 24 may issue the final optimized performance hard stop.

## Method

Use fixed, known candidate tokens and tree topologies of 1, 5, 15, 31 and 63 nodes. Include chains, balanced trees and one dynamic topology matched by node count.

For every topology/context:

- verify exact target logits against individually unrolled target calls;
- time Mamba layers, GQA layers, MoE layers, LM-head and scheduling separately;
- record expert union, H2D, VRAM and temporary recurrent/KV states;
- fit `T(N)=C+aN+bN^2`, but keep raw results authoritative;
- report warm and cold runs.

## Forbidden

- No target-informed acceptance claims here.
- No drafter.
- No component-time sum as a substitute for measured full verifier time.

## Deliverables

- preregistration;
- raw per-tree traces;
- exactness report;
- `P1_TREE_ROOFLINE.json`;
- roofline plot and terminal verdict.

## V2 claim boundary

A baseline ceiling below 250 tok/s does not close the program while the exact-efficiency track is open. Preserve it as the before-state for Agent 24.
