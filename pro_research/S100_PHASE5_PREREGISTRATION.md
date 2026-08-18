# S100 phase 5 — selective routed-work compiler

Date frozen: 2026-08-17
Parent evidence: phase-4 one-click v3.

## Frozen starting point

QFAST is the only phase-4 approximate primitive promoted to full fidelity:
18.75165 ms/token, 53.3286 tok/s, 10,240-target full fidelity green.
Global Mamba-W4, K5 and K4 are not adopted.

## Purpose

Recover part of the global K5/K4 speedup without paying their global quality
cost, and test whether small ReLU² values can be removed from the routed-down
path more gently than dropping whole experts.

## Three-way data split

The existing 40-prompt frozen phase-3 manifest is partitioned before phase-5
measurement:

- calibration: every `_01` prompt, 10 prompts, first 64 targets each;
- validation: every `_02` prompt, 10 prompts, first 128 targets each;
- heldout: every `_03` and `_04` prompt, 20 prompts, all 256 targets each.

No heldout metric may affect layer ranking, K-map construction, threshold
selection or candidate creation.

## Layer-K calibration

There are 23 MoE layers. On QFAST, evaluate one changed layer at a time:

- K6 -> K5;
- K6 -> K4.

The calibration quality cost is frozen as:

`cost = max(0, dKL) + 0.25*max(0, dCE) + 0.20*max(0, dTop1)`

where deltas are relative to QFAST calibration metrics. K6->K5 is the first
expert-drop action for a layer. K5->K4 uses the incremental K4-minus-K5 cost.
A greedy precedence-constrained ranking chooses the least-cost eligible action.

Frozen total routed-slot drop budgets are 4, 8, 12, 16, 20 and 24 across the
23 layers. Each budget produces one static per-layer K map with K in {4,5,6}.

## ReLU² threshold calibration

The custom panel scan computes the per-routed-expert max activation inside the
existing one-block scan and ignores a value only for routed-down execution when:

`0 < act < alpha * max(act)`.

Up projection, routing, route weights and shared expert remain unchanged.
Frozen alpha values:

- 0.0001
- 0.0003
- 0.0010
- 0.0030

Alpha=0 is an exact-QFAST control and must reproduce QFAST masks/semantics.

## Validation selection

All six static K maps and all four threshold arms are evaluated on the validation
split. The largest K-drop budget passing every strict validation gate is selected.
The largest alpha passing every strict validation gate is selected.

Strict validation gates:

- top1 agreement >= 0.970;
- exact V18 target in candidate top5 >= 0.999;
- mean CE delta <= 0.025 nat;
- mean coarse KL <= 0.015;
- p95 coarse KL <= 0.060;
- every domain top1 >= 0.90;
- every domain mean CE delta <= 0.080;
- finite outputs.

The selected K map, selected threshold and their combination are frozen before
heldout is read. If no nonzero arm passes, that candidate family is omitted.

## Heldout gates

The frozen phase-3 gates are reused on the untouched 5,120-target heldout split:

- top1 >= 0.95;
- target in top5 >= 0.995;
- mean CE delta <= 0.05;
- p95 CE delta <= 0.25;
- mean coarse KL <= 0.02;
- p95 coarse KL <= 0.08;
- every domain top1 >= 0.90;
- every domain mean CE delta <= 0.10;
- finite;
- deterministic anchor repeat.

The already proven K1 destructive control from phase 4 remains the harness-power
control and is not rerun for every candidate.

## Timing

Only heldout-green candidates get timing. Every candidate is compared with a
fresh-process QFAST base:

BASE_QFAST_A -> CAND_A -> CAND_B -> BASE_QFAST_B

All four are separate Python processes. Full timing uses 765 samples per arm.
Drift gates are <=1.0 ms for both base and candidate. VRAM <=7.8 GiB.

## Claim boundary

A phase-5 candidate is a compiled QFAST-derived model, not bit-identical V18.
S100-single still requires <=10.000 ms/useful token and a green heldout fidelity
result. Calibration or validation performance never counts as a final claim.
