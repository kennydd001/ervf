# Agent 04 — Target-Informed Proposal Coverage Oracle

## Mission

Determine whether the target distribution admits a small tree that covers a deep realized path. This is an upper bound, not a deployable drafter.

## Method

On validation only, construct optimal/near-optimal trees with node budgets 8, 16, 32 and 64 using target probabilities. Compare:

- chain;
- top-k fixed tree;
- Sequoia-style DP;
- OPT-Tree expected-acceptance objective;
- DySpec-style greedy expansion;
- sampling-without-replacement diversity;
- physical-cost-aware objective using measured verifier costs.

For greedy decode, measure the depth of the target greedy path contained in the tree. For sampling, use the exact registered acceptance algorithm.

## Primary gates

- depth >=7 with <=32 nodes; or
- depth >=9 with <=64 nodes.

## Hard boundary

The target-informed tree may never be described as a real drafter. It only establishes an information/coverage ceiling.
