# NEMOTRON TreeSweep-200 + Exact Roofline V2 Agent Pack

## Purpose

This pack tests whether **single-stream 200 token/s** is physically and statistically attainable for the user's Nemotron 3.5 Lightning/Nemotron-hybrid runtime on an RTX PRO 2000 Blackwell Laptop GPU with 8 GiB VRAM.

The central program is not "make the current decoder 7x faster." It is:

1. generate a compact tree of plausible continuations;
2. compile that tree into one exact hybrid target pass;
3. share target weights, Mamba state transitions, GQA/KV prefix work, and MoE expert records across all branches;
4. commit only the target-approved prefix;
5. optionally reduce exact target bytes through OrbitANS or CertiPlane;
6. keep any lossy PathQ target in a separate quality-controlled registry.

The primary system is called **TreeSweep-200**. The proposed scientific formulation is **Causal Branch Compilation (CBC)**: compile a candidate tree into the cheapest exact physical execution graph for a hybrid Mamba–Transformer MoE target.


## N1–N5 roofline pivot

A new exact-efficiency track has been added from five user-supplied measurements that must be independently reproduced:

- measured device streaming roofline around 338.4 GB/s;
- 23.7% token-time reduction from CUDA graph execution;
- an 8.192 ms sparse-down gather bottleneck;
- attention at about 47.2 GB/s with byte-linear R²=0.9964;
- critical GEMV at about 81.4 GB/s;
- exact low-rank ReLU² pre-gating closed as negative.

The pack now separates three milestones:

1. **E50 exact efficiency** — 50 tok/s without speculation;
2. **E100 exact efficiency stretch** — near-roofline 100 tok/s at short context;
3. **TreeSweep-200** — temporal amortization remains mandatory because the measured one-token byte floor caps a one-output sweep below 200 tok/s.

Read `ROOFLINE_PIVOT_N1_N5_REPORT_2026-08-15.md` before running the original tree oracles. The original unoptimized verifier is now diagnostic; only the optimized rerun may hard-close the 200 tok/s performance track.

## Core arithmetic

Provisional measurements from the existing EXACTFLOW work must be reverified in P0:

- short AR: 27.743 tok/s = 36.045 ms/token;
- average accepted draft tokens: 2.114;
- average output tokens per target round: 3.114.

For 200 tok/s, a round with `A` output tokens may cost at most:

```text
A=3.114  -> 15.57 ms
A=5      -> 25.00 ms
A=8      -> 40.00 ms
A=10     -> 50.00 ms
```

A diagnostic homogeneous-chain fit gives conditional acceptance `p≈0.7258`. Under that approximation, an infinitely long chain saturates near 2.647 accepted drafts, or 3.647 total output tokens per round. Therefore a longer linear MTP chain alone is not a credible 200 tok/s strategy.

## Mandatory execution order

1. Read `SHARED_RESEARCH_CONTRACT.md`.
2. Read `TREESWEEP_200_BREAKTHROUGH_REPORT_2026-08-15.md`.
3. Read `PAPER_MAP_2026-08-15.md`.
4. Read `EXPERIMENT_REGISTRY.yaml`.
5. Read `ROOFLINE_PIVOT_N1_N5_REPORT_2026-08-15.md`.
6. Run `00_MASTER_ORCHESTRATOR.md`.
7. Do not build a new drafter until the target-only verifier roofline and proposal coverage oracle pass.

## Highest-value early tests

- **P1 Target-only verifier roofline**: Can 15/31/63-node candidate trees be verified in one hybrid target call with exact outputs?
- **P2 Oracle proposal coverage**: With an ideal target-informed tree, can depth 7 be covered with ≤32 nodes or depth 9 with ≤64 nodes?
- **Joint speed-of-light gate**: Does any node budget yield at least 250 target-only committed-equivalent tok/s, leaving room for draft overhead?

If proposal coverage fails, the current tree family is closed. If the **optimized** joint oracle ceiling after the exact-efficiency track misses the 250 tok/s gate, single-stream 200 tok/s is falsified for this target/hardware pair. The unoptimized baseline verifier is diagnostic only.

## Pack contents

- `00_MASTER_ORCHESTRATOR.md` — execution controller.
- `SHARED_RESEARCH_CONTRACT.md` — scientific rules.
- `TREESWEEP_200_BREAKTHROUGH_REPORT_2026-08-15.md` — full hypothesis report.
- `PAPER_MAP_2026-08-15.md` — primary-paper synthesis and prior-art boundaries.
- `PAPER_MATRIX.csv` — machine-readable paper map.
- `EXPERIMENT_REGISTRY.yaml` — dependencies, gates and stop rules.
- `CURRENT_MEASUREMENTS.template.json` — provisional identity/MTP schema.
- `CURRENT_MEASUREMENTS.N1_N5_IMPORTED.json` — imported roofline evidence awaiting reproduction.
- `ROOFLINE_PIVOT_N1_N5_REPORT_2026-08-15.md` — exact-efficiency synthesis and new hypotheses.
- `agents/` — 24 specialized agent prompts, including the exact-efficiency track.
- `tools/` — transparent arithmetic and registry validation.
- `templates/` — preregistration and final-verdict templates.

## Claim boundary

This pack is a falsifiable research program, not a speed claim. A `verified_breakthrough` requires one integrated physical run, exact target semantics or a separately registered quality gate, at least 512 causal output tokens, p95/p99 reporting, thermal stability, and independent verification.
