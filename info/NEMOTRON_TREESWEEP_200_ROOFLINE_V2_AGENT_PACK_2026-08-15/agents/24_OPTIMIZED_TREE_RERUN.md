# Agent 24 — Optimized Target-Tree Roofline Rerun

## Mission

Rerun the target-only tree verifier and joint speed-of-light calculation using the frozen best exact runtime from Agent 23 or, if Agent 23 misses its gate, the best independently verified exact component set.

## Rules

- proposal coverage data are unchanged;
- no new drafter;
- natural target routing and exact target semantics;
- same tree node budgets and topologies as the baseline roofline;
- report baseline versus optimized verifier side by side;
- only this optimized joint ceiling can hard-falsify the 200 tok/s exact track on performance grounds.

## Hard gate

```text
max_N oracle_output_tokens(N) / optimized_verifier_time(N) >= 250 tok/s
```

If the gate fails after coverage and exact-efficiency branches are frozen, close the 200 tok/s exact track for this target/hardware pair.
