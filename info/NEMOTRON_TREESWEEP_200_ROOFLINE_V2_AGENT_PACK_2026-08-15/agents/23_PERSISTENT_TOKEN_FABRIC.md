# Agent 23 — Exact Persistent Token Fabric

## Mission

Integrate the best frozen exact candidates from Agents 19–22 into one physical decoder and test 50/75/100 tok/s milestones. Do not sum projected percentages.

## Candidate structure

- full-token graph or device-resident graph loop;
- fixed buffers and no per-token allocation;
- gather-free downflow;
- roofline-oriented attention and GEMV;
- device-resident token sampling and state update where exact;
- dynamic route/expert IDs as data, not graph topology;
- one integrated timing domain.

## Required milestones

- E50: ≤20.000 ms/token;
- E75: ≤13.333 ms/token;
- E100: ≤10.000 ms/token.

## Final E50 proof

- ≥10,000 causal output tokens across prompt domains;
- contexts 0/4K/32K and separate ~262K profile;
- exact target semantics;
- p50/p95/p99/max;
- 8 GiB VRAM and registered RAM budget;
- no swap;
- one-hour thermal run;
- independent verifier.

## Relationship to TreeSweep

A failed E50 does not automatically falsify 200 tok/s. The best frozen exact runtime is passed to Agent 24 and the optimized TreeSweep roofline. A successful E50 lowers the verifier round time but 200 still requires temporal amortization because the one-token byte floor is >5 ms.
