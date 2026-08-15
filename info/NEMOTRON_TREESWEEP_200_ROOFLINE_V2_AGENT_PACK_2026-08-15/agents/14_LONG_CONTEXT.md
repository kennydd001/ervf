# Agent 14 — Long-Context TreeSweep

## Mission

Preserve speculative advantage at 128K and 262K context.

## Candidate mechanisms

- LongSpec constant-memory draft KV and hybrid prefix/tree attention;
- QuantSpec hierarchical quantized draft KV;
- SparseSpec sparse draft attention and delayed verification;
- page-aware GQA tree attention;
- context-dependent tree-size controller.

## Measurements

For 4K, 32K, 128K and 262K:

- draft time;
- tree verifier time;
- KV bytes and bandwidth;
- accepted output per round;
- memory high-watermark;
- target quality.

## Gates

- >=100 tok/s at 128K;
- >=60 tok/s at 262K;
- no quality regression on exact track;
- memory gate.
