# S100 phase 3 V18-fidelity preregistration

Date frozen: 2026-08-17

## Purpose and boundary

This suite answers whether an approximate runtime preserves exact V18's output
distribution on a frozen multi-domain V18-generated trajectory. It is a
self-distillation/fidelity test, not ground-truth corpus perplexity and not a
substitute for external benchmarks.

The exact V18 trace is produced once and hashed. Candidate runs may not alter or
regenerate it after seeing candidate results.

## Data

`S100_PHASE3_PROMPTS.json` contains 40 authored prompts in 10 declared domains:
English and Dutch factual continuation, reasoning, mathematics, code,
hardware/LLM engineering, English and Dutch creative writing, summarisation and
instruction following.

- smoke: first 8 prompts, 64 target tokens each = 512 tokens;
- full: all 40 prompts, 256 target tokens each = 10,240 tokens.

For every exact-V18 target position, the trace stores:

- exact greedy target id;
- exact target log probability;
- exact top-64 ids and log probabilities;
- probability mass outside that top-64.

## Candidate profiles

- `qfast`, `fast`, `k5`, `k4`, `fast_k5`, `fast_k4`;
- `k1_control` is intentionally destructive and must fail at least one fidelity
  gate, proving the harness has power.

## Teacher-forced metrics

Candidates are fed the exact V18 target tokens. Report:

- top-1 agreement;
- rank of V18's target under the candidate;
- V18 target contained in candidate top-5;
- top-5 overlap;
- candidate CE on V18's target and delta from V18 self-CE;
- a 65-bucket coarse KL: V18 top-64 tokens individually plus one residual-mass
  bucket. This is a data-processing lower bound on full KL, not full KL;
- domain-stratified metrics;
- deterministic repeat hashes;
- finite-value checks.

## Frozen fidelity gates

- overall top-1 agreement >=0.950;
- V18 target in candidate top-5 >=0.995;
- mean CE delta <=0.050 nat;
- bootstrap 95% upper bound of mean CE delta <=0.075 nat;
- p95 CE delta <=0.250 nat;
- mean coarse KL <=0.020;
- p95 coarse KL <=0.080;
- every domain top-1 agreement >=0.900;
- every domain mean CE delta <=0.100 nat;
- deterministic anchor repeat;
- all logits/metrics finite;
- k1_control must fail at least one gate.

## Greedy trajectories

The candidate also generates independent greedy continuations. First divergence,
position-wise token agreement and decoded side-by-side text are reported for
manual/agent review, but are not converted into a scalar quality claim.

A passing candidate becomes `v18_fidelity_candidate`. External task and
perplexity evaluation remains mandatory before adoption.
