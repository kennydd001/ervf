# Agent 07 — Exact MoE Expert-Union Compiler

## Mission

Execute natural target MoE routes for all tree nodes while loading/decoding each expert record once per verification round where possible.

## Method

Per layer:

1. route every node with the target router;
2. collect `(expert_id, node_id, router_weight)` tuples;
3. sort/group by expert;
4. load or reuse each expert once;
5. process all assigned node rows;
6. apply original router weights and scatter.

## Tree-cost model

Fit measured cost from:

- union size;
- cache residency;
- rows per expert;
- H2D bytes;
- grouped-kernel efficiency;
- overlap slack.

Use the model only for tree selection; final routing stays exact.

## Required baselines

- node-count-only tree;
- acceptance-only tree;
- EcoSpec-like marginal expert-cost tree;
- exact physical-cost tree.

## Stop

If expert union destroys the joint oracle advantage, do not drop experts inside the exact registry. Close or redesign the candidate tree.
