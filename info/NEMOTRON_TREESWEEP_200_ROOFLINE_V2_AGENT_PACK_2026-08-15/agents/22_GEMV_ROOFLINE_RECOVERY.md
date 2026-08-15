# Agent 22 — Critical GEMV Roofline Recovery

## Mission

Optimize the complete weighted critical-shape GEMV suite, not a cherry-picked inner kernel, from the imported ~81.4 GB/s toward the measured streaming roofline.

## Candidate mechanisms

- activation-tile broadcast across output-row subwarps;
- scale broadcast and vectorized code loads;
- persistent row work queues;
- exact virtual reduction-width autotuning;
- producer/consumer weight-load pipeline;
- same-input projection families with shared activation load;
- fixed workspace and graph-resident pointers;
- projection-specific kernels rather than one universal kernel.

## Required counters

- bytes and transactions;
- achieved occupancy;
- register/local-memory use;
- branch divergence;
- instruction mix;
- L1/L2 hit rate;
- per-shape and weighted-suite results.

## Gates

- reproduce ~81.4 GB/s;
- first gate weighted suite ≥140 GB/s;
- strong gate ≥170 GB/s;
- no critical shape >5% slower;
- integrated token improvement ≥8% before this branch is called useful.
