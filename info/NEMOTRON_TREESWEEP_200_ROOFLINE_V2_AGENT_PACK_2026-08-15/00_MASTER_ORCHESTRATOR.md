# MASTER ORCHESTRATOR — TreeSweep-200 + Exact Roofline V2

You are the lead research agent. Your task is to test TreeSweep-200 rigorously and preserve negative results.

## Mandatory reads

1. `README.md`
2. `SHARED_RESEARCH_CONTRACT.md`
3. `TREESWEEP_200_BREAKTHROUGH_REPORT_2026-08-15.md`
4. `PAPER_MAP_2026-08-15.md`
5. `ROOFLINE_PIVOT_N1_N5_REPORT_2026-08-15.md`
6. `EXPERIMENT_REGISTRY.yaml`
7. `CURRENT_MEASUREMENTS.json` and `CURRENT_MEASUREMENTS.N1_N5_IMPORTED.json`

## Mandatory isolation

Use only:

```text
reports/treesweep200/
scripts/treesweep200/
src/treesweep200/
tests/treesweep200/
```

Do not modify any closed registry.

## Phase order

### Phase 0 — identity and baseline

Run `agents/01_IDENTITY_BASELINE.md`.

No other phase opens until target identity, MTP identity, hashes, runtime commit and provisional measurements are locked.

### Phase E0 — reproduce N1–N5 before using them

Run `agents/18_ROOFLINE_REPRODUCTION.md`. Imported values remain `USER_MEASURED_UNVERIFIED` until their raw sources, hashes and reproduction verdicts close. N3 low-rank prefiltering remains closed unless E0 finds a data or implementation error.

### Phase 1 — coverage, baseline verifier and exact-efficiency track

Run after P0/E0:

- `agents/02_ROOFLINE_ORACLE.md` — baseline verifier measurement only;
- `agents/03_TARGET_TREE_VERIFIER.md`;
- `agents/04_PROPOSAL_COVERAGE_ORACLE.md`;
- `agents/19_GRAPH_RESIDENT_TOKEN.md`;
- `agents/20_GATHERLESS_DOWNFLOW.md`;
- `agents/21_ATTENTION_ROOFLINE_RECOVERY.md`;
- `agents/22_GEMV_ROOFLINE_RECOVERY.md`.

Coverage may hard-stop immediately because it is algorithmic. A slow baseline verifier may **not** hard-stop the 200 tok/s track after N1–N5. Freeze the best exact candidates and run `agents/23_PERSISTENT_TOKEN_FABRIC.md`, then `agents/24_OPTIMIZED_TREE_RERUN.md`.

Final performance hard stop:

```text
max_N oracle_output_tokens(N) / optimized_verifier_time(N) < 250 tok/s
```

Only the optimized verifier plus frozen coverage oracle may close the exact 200 tok/s performance track.

### Phase 2 — exact hybrid tree kernels after optimized joint gate

Only after the joint oracle passes:

- `05_MAMBA_TREE_SCAN.md`
- `06_TREE_ATTENTION.md`
- `07_MOE_EXPERT_UNION.md`

Each has an independent exactness and speed gate.

### Phase 3 — cheapest real drafter

Run `08_NATIVE_MTP_TREE.md` first. Do not train a new drafter unless native MTP tree coverage/output-ms misses the gate.

### Phase 4 — learned drafter branches

Conditional alternatives:

- `09_PARALLEL_DRAFTER.md`
- `10_DIFFUSION_DRAFTER.md`

They must share the same frozen target verifier and validation/test protocol.

### Phase 5 — optional exact/quality modifiers

Only after a measured bottleneck:

- `11_BRANCHCERT.md`
- `12_ORBITANS_PATHQ.md`
- `13_ASYNC_HETEROGENEOUS.md`
- `14_LONG_CONTEXT.md`

No modifier may be added merely because it has a positive component benchmark.

### Phase 6 — integrated proof

The exact-efficiency track may independently verify E50/E75/E100, but only temporal amortization can verify E200 because the measured one-token byte floor exceeds 5 ms.

Run `15_INTEGRATION_200.md`.

Milestones:

1. 50 tok/s exact;
2. 100 tok/s exact;
3. 200 tok/s exact.

Only the third plus all memory, rollout, thermal and verification gates yields `verified_breakthrough`.

### Phase 7 — audits

- `16_NOVELTY_AUDIT.md`
- `17_INDEPENDENT_VERIFIER.md`

## Required verdict vocabulary

Use only the statuses from the shared contract. Never use “Eureka” inside the repository before the integrated verifier marks `verified_breakthrough`.
