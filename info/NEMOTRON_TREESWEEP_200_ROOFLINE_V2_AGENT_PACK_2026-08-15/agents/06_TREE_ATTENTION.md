# Agent 06 — GQA Tree Attention

## Mission

Implement topology-aware attention for the six GQA layers with shared prefix KV and branch-local temporary KV.

## Required variants

1. unrolled branch batch;
2. packed topology mask;
3. LongSpec-style split between optimized prefix attention and tree-mask attention;
4. optional page-aware GQA tile sharing.

## Controls

- every node attends only to permanent prefix plus ancestors;
- siblings never leak;
- accepted KV commit equals sequential target;
- rejection does not mutate permanent cache.

## Gates

At 4K the packed variant must beat unrolled attention. Long-context gates are handled separately by Agent 14.
