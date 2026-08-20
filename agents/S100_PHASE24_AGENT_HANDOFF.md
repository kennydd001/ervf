# S100 Phase24 handoff

## What is already combined

The adopted Phase23R graph already includes:
- exact Lightning target math;
- Phase19 FP8 residual2 Mamba projections;
- Phase17 affine Mamba block recurrence;
- Phase22 production_x4 lm_head;
- V6 device cache/routing foundations;
- H4 CUDA graph;
- GPU-resident grouped routed experts.

Do not describe Phase23R as a separate 49 tok/s product replacing V18. It is a
perfect-draft verifier experiment.

## What remains to synthesize

V18's principal missing feature is H-SCALE:
resident per-cache-slot down-projection scale planes, removing panel-scale
metadata from PCIe gathers. Full allocation is ~492 MiB.

Phase24 actual-allocates a ranked layer ladder. No `mem_info` admission test.

Dense exact batching:
- attention BF16 q/k/v/o M4 loads each weight once for four rows and reproduces
  the production GEMV thread/reduction order;
- router FP32 M4 does the same;
- shared expert M4 is screened against four production NVFP4 calls before use.

## Important non-assumption

Phase23 already has grouped_up_m1. “Use V18 M1” is not a direct function swap.
The mature V18 single-token `_moe_dev` computes all six routes of one token and
cannot process an isolated M1 expert group without new scheduling.

The clean compatible V18 contribution is resident scale data placement and
copy/gather overlap.

## Structural second axis

Even a perfect H4 synthesis may not halve 81.9 ms to <40 ms. Phase24 therefore
records H8 route multiplicity across the same tokens and across eight prompt
classes. If H4 remains above 40 ms and H8 adds meaningful sublinear expert
growth, Phase25 H8/H16 is opened automatically.
