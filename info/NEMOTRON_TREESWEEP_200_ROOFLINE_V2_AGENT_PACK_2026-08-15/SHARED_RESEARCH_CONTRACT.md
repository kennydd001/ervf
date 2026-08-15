# Shared Research Contract — TreeSweep-200

## 1. Isolation

Use a fresh namespace:

```text
reports/treesweep200/
scripts/treesweep200/
src/treesweep200/
tests/treesweep200/
```

Do not modify or reinterpret CRAFT, RSIV/GhostWeights, BITFLOW, CORETAIL, STREAMQ5, PORT80B, EXACTFLOW, or any other closed registry.

## 2. Source labels

Every claim must be tagged as one of:

- `OFFICIAL_SOURCE`
- `PRIMARY_PAPER_CLAIM`
- `USER_MEASURED_VERIFIED`
- `USER_MEASURED_UNVERIFIED`
- `OWN_CALCULATION`
- `HYPOTHESIS`
- `PROJECTION_ONLY`

## 3. Identity first

The user's runtime is called “Nemotron 3.5 Lightning,” but public Nemotron artifacts differ in routed/shared expert counts and do not by themselves prove the measured MTP path. P0 must lock:

- model ID and revision;
- all artifact hashes;
- tokenizer/chat template;
- routed and shared expert counts;
- exact MTP implementation and weights;
- runtime commit and compiler flags;
- whether the target is public Nano, NIM-only, Elastic, custom, or a derived artifact.

No paper/model claim may be transferred across identities without an explicit equivalence test.

## 4. Exact versus quality-changing tracks

### Exact track

TreeSweep, STree-like Mamba tree scan, tree attention, exact natural MoE routing, OrbitANS, CertiPlane and exact asynchronous scheduling must preserve the frozen target distribution.

For greedy decoding, every emitted token must match the target. For stochastic decoding, use a distribution-preserving verification rule and validate its statistical implementation.

### Quality track

PathQ, Nested-QAD, diffusion target replacement, expert budgeting, route dropping or lossy verification create a new target. They require an independent quality registry and may never be described as lossless.

## 5. Validation/test discipline

- Choose configurations only on validation.
- Freeze the candidate and hashes before opening test.
- Open test once per registered candidate.
- Failed gates remain failed.
- Use prompts/sequences as bootstrap units, not individual correlated tokens.

## 6. No multiplication of component speedups

A combined speed claim requires one integrated physical run. Component speedups may appear only as labeled projections.

## 7. Oracle order

Before training or writing a complex runtime:

1. target-only tree verifier roofline;
2. target-informed proposal-tree coverage oracle;
3. joint oracle throughput ceiling;
4. only then a real drafter.

An oracle that misses its gate blocks all dependent implementation.


## 7A. Exact-efficiency roofline track

The N1–N5 branch changes no model semantics. It attempts to execute compulsory work closer to the measured hardware roofline through CUDA graphs, gather-free downflow, attention/GEMV kernel redesign and persistent scheduling.

Rules:

- imported measurements are unverified until Agent 18 closes;
- exact single-token milestones are E50/E75/E100;
- no component percentages may be summed;
- the integrated Agent 23 result is authoritative;
- the low-rank ReLU² prefilter is closed negative and cannot be reopened by a larger rank sweep;
- 200 tok/s still requires temporal amortization because the measured one-token byte floor is above 5 ms.

## 8. Student-state and causal requirements

Any model-changing method must run full-depth on its own states. Teacher-forced local reconstruction is not evidence of stable generation.

The integrated gate requires:

- at least 512 causal output tokens per prompt set;
- independent target and candidate prefix states;
- exact state commit after speculative rejection;
- no hidden teacher-state injection.

## 9. Hybrid-state correctness

Nemotron includes Mamba-2, GQA and MoE components. Verification must separately control:

- Mamba recurrent states and activation replay;
- GQA tree masks and KV commit;
- natural target routing for every tree node;
- expert-major batching without route modification;
- LM-head and acceptance semantics.

## 10. Hardware accounting

Report at minimum:

- total round time and draft/verify split;
- target verifier time by tree size and depth;
- Mamba, GQA, MoE, head and scheduling time;
- expert union and H2D bytes per layer/round;
- VRAM high-watermark;
- process RSS/commit and swap/page faults;
- p50/p95/p99/max latency;
- clocks, temperature and thermal decay;
- context length and batch size.

## 11. Primary 200 tok/s definition

The primary claim is **single-stream**, batch 1, short-context decode:

```text
>= 200 output tok/s
8 GiB VRAM
64 GiB research RAM; 32 GiB product stretch gate
>= 512 causal output tokens
exact target semantics on the exact track
p95/p99 reported
one-hour thermal run before final claim
independent verifier
```

Aggregated multi-user throughput must be reported separately.

## 12. Stop rules

- If target-informed coverage fails its frozen gate, close the current tree family.
- If the **optimized** target verifier from Agent 24 plus frozen target-informed coverage cannot exceed 250 committed-equivalent tok/s at any node budget ≤64, close the 200 tok/s exact track. The unoptimized baseline verifier is diagnostic only.
- If depth 7 cannot be covered with ≤32 nodes and depth 9 cannot be covered with ≤64 nodes in the oracle, close the current proposal-tree family.
- If STree-style state scan or tree attention is slower than unrolled verification at the registered tree budgets, close that kernel branch.
- If MoE expert-union traffic erases the verifier advantage, do not drop experts inside the exact registry; redesign the tree or close the branch.
- If a learned drafter cannot beat the frozen native MTP baseline on integrated output/ms, stop training it.
- If exact certificate pass rate is below its gate, do not replace it with an unsafe learned risk predictor inside the exact registry.

## 13. Verdict vocabulary

Use only:

- `queued`
- `dependency_blocked`
- `running`
- `screen_positive`
- `screen_negative`
- `gate_passed`
- `gate_failed`
- `falsified`
- `inconclusive`
- `verified_breakthrough`
