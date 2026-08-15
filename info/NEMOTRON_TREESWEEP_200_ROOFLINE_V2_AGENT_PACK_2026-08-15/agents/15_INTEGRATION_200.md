# Agent 15 — Integrated 200 tok/s Proof

## Mission

Integrate exactly one frozen candidate. No projected speedups count.

## Required stages

1. 50 tok/s milestone;
2. 100 tok/s milestone;
3. 200 tok/s breakthrough run.

## Primary run

- batch 1;
- short context frozen by P0;
- at least 512 causal output tokens per prompt set;
- multiple domains, including code/reasoning/general text;
- 8 GiB VRAM;
- 64 GiB research RAM and separately report 32 GiB feasibility;
- no swap;
- exact target semantics or registered quality target;
- p50/p95/p99/max;
- TTFT/prefill separate;
- one-hour thermal steady state for final claim.

## Round telemetry

For every round record:

- candidate node count/depth;
- accepted depth;
- draft ms;
- verifier ms;
- Mamba/GQA/MoE/head ms;
- expert union and H2D;
- state/KV commit cost.

## Verdict

Only all registered gates plus independent verification permit `verified_breakthrough`.
