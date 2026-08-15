# Agent 03 — Exact Hybrid Target Tree Verifier

## Mission

Build the minimal correct tree-verification path that processes a packed candidate tree in one target invocation and commits only the accepted prefix.

## Scope

- exact Mamba/GQA/MoE target semantics;
- greedy verification first;
- distribution-preserving sampling only after greedy controls close;
- separate cache/state accounting from semantic commit.

## Controls

For every node compare:

- logits;
- routed expert IDs and weights;
- Mamba temporary outputs;
- GQA outputs;
- accepted token path;
- final committed KV and Mamba states.

Compare against separate sequential target calls for all root-to-leaf paths.

## Gates

- exact tokens and accepted states;
- no sibling leakage;
- no unregistered approximation;
- full measured verifier times supplied to the roofline oracle.
