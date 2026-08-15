# Agent 16 — Novelty and Claims Audit

## Mission

Determine what, if anything, is technically new after a candidate passes.

## Mandatory comparison families

- token-tree verification: SpecInfer, Sequoia, OPT-Tree, DySpec;
- SSM/hybrid tree scan: STree, speculative Mamba;
- draft heads: Medusa, Hydra, EAGLE-3, FastEagle, PTP;
- long-context speculation: LongSpec, SparseSpec, QuantSpec;
- MoE speculative cost: MoE-Spec, EcoSpec, AcceptMoE, EdgeXpert;
- diffusion draft: DEER, D2SD, PRESTO, Nemotron-TwoTower;
- async/heterogeneous: SwiftSpec, AHASD, Dovetail;
- exact precision/entropy branches from EXACTFLOW.

## Claim units

Audit separately:

1. Causal Branch Compilation objective;
2. exact hybrid Mamba/GQA/MoE tree verifier;
3. physical expert-union-aware tree optimizer with natural routing;
4. BranchCert off-path precision;
5. integrated consumer 8 GiB/200 tok/s result.

A non-found exact combination is negative search evidence, not proof of novelty or patentability.
