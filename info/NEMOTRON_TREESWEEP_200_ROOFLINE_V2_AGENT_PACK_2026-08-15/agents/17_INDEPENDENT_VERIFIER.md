# Agent 17 — Independent Verifier

## Mission

Recompute every final gate from immutable raw artifacts without importing the candidate implementation's metric code.

## Required checks

- hashes and revisions;
- registry dependency order;
- test opened only after frozen validation choice;
- exact target token/distribution checks;
- Mamba and KV accepted-state digests;
- natural MoE route and router-weight controls;
- round-time arithmetic and tok/s;
- p50/p95/p99/max;
- VRAM/RAM/swap;
- 512-token and thermal gates;
- no component-speedup multiplication;
- N1–N5 imported evidence hashes and reproduction status;
- byte-floor arithmetic and decimal/binary unit consistency;
- graph/eager identical-work control;
- gather-free support, code/scale and reduction-graph controls;
- attention/GEMV achieved-bandwidth arithmetic;
- E50/E75/E100 integrated milestones;
- baseline versus optimized tree-roofline separation;
- exact vs quality claim separation.

## Output

- machine-readable verification JSON;
- readable report;
- total checks/pass/fail/warnings;
- final status, which alone may be `verified_breakthrough`.
