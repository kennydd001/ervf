# Agent 19 — Graph-Resident Token Program

## Mission

Eliminate repeated host/driver work without changing target bytes or math.

## Variants

1. eager reference;
2. stream-captured full-token CUDA graph;
3. explicitly constructed graph with stable addresses;
4. context-bucket graph family;
5. graph with dynamic routes through device-resident indirection arrays;
6. device conditional/tail-launch token loop, only when supported by the exact CUDA/runtime build.

## Dynamic-data rule

Graph topology must remain fixed. Token ID, KV position, top-k IDs, cache slots and pointers are data consumed by fixed kernels; they must not cause per-token graph reconstruction.

## Controls

- exact logits/tokens/routes/Mamba/KV state versus eager;
- same bytes and kernels for the first capture comparison;
- graph build and update time reported separately;
- no hidden synchronization or host callback inside the timed region;
- 10,000-token stability test for final gate.

## Gates

- reproduce the imported ~23.7% improvement within 5%;
- strong standalone gate: token time ≤26 ms before other new kernels;
- p95 improves, not only mean;
- no forbidden extra VRAM allocation.
