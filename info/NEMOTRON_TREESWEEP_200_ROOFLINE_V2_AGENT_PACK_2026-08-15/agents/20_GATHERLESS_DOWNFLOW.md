# Agent 20 — Gather-Free Sparse Downflow

## Mission

Remove the global sparse-activation gather while preserving the exact post-ReLU² mask and down-projection semantics.

## Reference path

```text
ReLU² output → panel scan → gather buffer → sparse down projection
```

## Candidate path

```text
ReLU² output + exact bitmask → direct index-carrying down projection
```

## Required variants

1. no-compaction direct masked down;
2. warp-ballot iterator over selected logical channels;
3. panel-major physical weight layout with virtual original group IDs;
4. fused scan+down;
5. fused ReLU²-mask producer and down consumer via shared/ring buffer;
6. persistent expert CTA if register/local-memory accounting permits.

## Exactness

The selected support is unchanged. Survivor codes, scales, logical channel IDs, dtype boundaries and registered reduction order must be controlled. Any different numerical target requires a separate quality registry.

## Instrumentation

- gather bytes and transactions;
- global-memory round trips;
- effective down-weight bandwidth;
- mask density and distribution;
- register spills/local memory;
- occupancy and warp stalls.

## Gates

- eliminate ≥80% of measured gather time;
- down path ≥1.8× faster;
- no quality/semantic change;
- integrated graph+gather-free candidate ≤20 ms/token at context 0 for the strong gate.
