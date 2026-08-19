# S100 Lightning Phase 17 — exact component-oracle atlas

Date: 2026-08-19
Parent: `agent/s100-lightning-phase16r-recovery-hardware` at `4345608`
Target: the same NVIDIA Nemotron 3.5 Lightning checkpoint and frozen quality parent.

## 17A result and claim boundary

The steady-state K/V replay oracle is parity-green after discarding the known first post-build outlier. Recorder, oracle, control A and control B produce identical tokens and final logits. The bracketed reference is about 19.236 ms/token; replacing the eleven heldout-green K/V projections by exact device-table replay saves about 0.036 ms/token, roughly 0.39% of the S100 latency gap.

This closes the eleven-matrix K/V route as a primary performance target, subject to two final reporting controls in the same result:

1. paired, prompt-clustered one-sided 95% confidence bound on bracket-corrected savings;
2. replay-copy overhead measured by executing the same replay copies into scratch while retaining the original K/V path.

The route is definitively closed at the 3% engineering gate when the corrected one-sided 95% upper bound remains below 3% of the parent latency.

## Why one-at-a-time timing is insufficient

Previous experiments showed that faster component streams can disappear end-to-end because work overlaps. A component may therefore save little alone yet become important when another overlapping component is removed.

Phase 17 measures a subset lattice, not only isolated arms. For oracle groups `M`, `E`, `A`, and `L`, measure all feasible singles and combinations. Report non-additivity:

`interaction(S,T) = saving(S union T) - saving(S) - saving(T)`.

Positive super-additivity identifies overlapping critical paths that cannot be diagnosed by isolated microbenchmarks.

## Frozen oracle groups

### L — LM-head ceiling

Teacher-forced target tokens make per-step logits semantically unnecessary. During timed positions, bypass the LM head and sampling/argmax while preserving the frozen target-token feed. Re-enable the original LM head on a final untimed position and require hidden state, recurrent state, KV state and final logits parity.

Report this explicitly as a teacher-forced LM-head upper bound, not as free-running achieved throughput.

### E — complete MoE ceiling

Record the exact output vector of every MoE layer for the frozen workload. In the oracle graph, replace routing, expert-cache operations, shared expert, routed-up, ReLU2, routed-down and route accumulation with an exact device replay of that layer output.

MoE is stateless with respect to model semantics, so hidden-state and final-logit parity are required. Cache/LRU metadata may differ and must be reported separately as intentionally bypassed performance state.

### A — complete attention-compute ceiling

Record per attention layer and position:

- exact K and V vectors to append;
- exact final attention-layer output before the residual add.

The oracle must append the recorded K/V values to the canonical FP8 KV cache at the normal position, replay the exact attention output and preserve all position accounting. Require used-KV bytes, hidden state, recurrent state and final logits to match the parent.

This group measures Q, K, V, attention core, O and related launches while retaining the unavoidable semantic KV-state update.

### M — Mamba dense-projection ceiling

Do not bypass Mamba state evolution. Replace only recorded dense projections while retaining convolution, selective SSM, gating, normalization and canonical state updates.

Measure separately:

- `M_in`: all Mamba in-projections;
- `M_out`: all Mamba out-projections;
- `M_io`: both.

Because full-table VRAM may be too large, partition Mamba layers into deterministic groups whose replay tables fit with at least 256 MiB free reserve. Run each group with independent A/O/B brackets and sum only after an additivity check on at least one adjacent group pair.

## Measurement protocol

- Lightning identity hash must equal the Phase-15/16 parent.
- Frozen ten-prompt teacher-forced workload.
- First complete workload pass discarded.
- Control A -> oracle -> control B, with an optional second oracle arm when thermal slope exceeds 0.15 ms.
- At least 560 measured positions per arm for small-table oracles; at least 256 for memory-heavy Mamba group oracles.
- Same target-token sequence in every arm.
- No host readback, allocation or synchronization inside captured graph execution.
- Every replay table is immutable, device resident and included in VRAM accounting.
- Record replay node count, replay bytes/token and copy-overhead control.

## Statistics

For each prompt/position, use the linearly bracketed reference between control A and B. Bootstrap by prompt, not by individual token, with 10,000 resamples.

Report:

- mean and median savings;
- one-sided 95% upper and lower confidence bounds;
- aggregate and p50 speedup;
- percentage of the 19.236 -> 10.000 ms S100 gap covered;
- thermal slope and A/B drift;
- replay-overhead-corrected savings.

## Decision bands

At the current ~19.2 ms parent:

- `>=0.58 ms` lower-confidence saving: immediate kernel/runtime target;
- `0.20-0.58 ms`: secondary target only when implementation is low risk or combines super-additively;
- `<0.20 ms` upper-confidence saving: close as an independent engineering route;
- combined oracle `<=10.0 ms`: an architecture-preserving S100 path exists in the measured component set;
- combined oracle `>10.0 ms`: the unmeasured remainder still requires a structural algorithmic change.

## Required subset lattice

Minimum arms:

- L, E, A;
- M_in, M_out, M_io;
- E+L, A+L, E+A;
- M_io+E, M_io+A, M_io+L;
- M_io+E+A+L where VRAM permits.

If a full combination does not fit, run two complementary table-resident combinations and report only an interval, never an additive point estimate.

## Next implementation rule

No new production kernel is built merely because its microbenchmark is faster. A component earns kernel work only after its parity-green oracle ceiling clears the decision band, or after the subset lattice proves material super-additivity.

## Publication

Commit the repaired 17A module, final JSON, statistical/copy-overhead addendum and the Phase-17 atlas implementation to this branch. Preserve Phase-16R results unchanged.
