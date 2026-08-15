# Agent 21 — Attention Roofline Recovery

## Mission

Raise exact decode attention from the imported ~47.2 GB/s toward the independently reproduced streaming roofline without first changing KV precision or model semantics.

## Required profiles

At 0/4K/32K/128K/~262K:

- raw contiguous KV scan;
- actual page/address scan without attention math;
- QK score;
- online softmax;
- PV accumulation;
- full attention;
- GQA head-group reuse;
- page-table/workqueue overhead.

## Candidate kernels

1. KV-head-group tiled decode: one K/V tile shared across all query heads in the group;
2. page-aware work queue and sequence splitting for batch-1 long context;
3. online softmax + value accumulation in one kernel;
4. vectorized/tiled K/V loads with double buffering;
5. producer/consumer warp specialization where supported;
6. context-specific autotuning;
7. CUDA-graph-compatible static workspace.

## Baselines

- current kernel;
- FlashInfer-compatible shape/config where possible;
- one high-quality reference implementation with identical KV format.

## Gates

- reproduce byte-linear fit with R²≥0.99;
- first gate ≥100 GB/s effective at long context;
- strong gate ≥169 GB/s;
- attention ≤6 ms at ~262K, stretch ~4.8 ms;
- identical KV mutation and controlled output error.
