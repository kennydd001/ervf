
# S100 phase 4 decision

## What phase 3 actually proved

- FAST remains a valid 17.38 ms performance candidate from phase 2.
- FAST is not V18-fidelity preserving: on the 512-token smoke trace it reached
  90.23% top-1 agreement, 98.44% target recall in top-5, +0.184 nat mean CE and
  0.151 mean coarse KL. Dutch factual prompts were substantially worse than
  English factual prompts.
- QFAST full timing was stable but only saved 0.580 ms/token.
- Every K-profile candidate repeated stably, but its in-process `BASE_B` did
  not return to the original performance state. The phase-3 midpoint savings
  are invalid.
- The phase-3 independent fidelity verifier failed only while serializing
  NumPy booleans. The underlying FAST and control result files remain usable.

## Immediate work

1. Run fresh-process timings for all profiles.
2. Complete smoke fidelity for QFAST, MAMBA, K5 and K4.
3. Run full fidelity only for smoke-green profiles.
4. Do not run FAST full fidelity; its smoke failure is not borderline.

## Quantitative reality

The fastest measured integrated candidate is still about 17.38 ms/token.
Even granting a valid K4 improvement of roughly one millisecond, another
6–7 ms must disappear to reach 10 ms. Weight-only conversion and global top-k
reduction cannot close that alone.

## Single-stream hypotheses after the repair

### H1 — selective layer portfolio

FAST changes 52 matrices at once. Its quality failure may be concentrated in a
small subset of Q or recurrent Mamba layers. Use a calibration trace to rank
individual or grouped conversions by:

- end-to-end CE/KL impact;
- greedy divergence;
- measured milliseconds saved.

Select a portfolio before evaluating a disjoint held-out trace. This is the
highest-priority quality-recovery hypothesis.

### H2 — layer-specific/adaptive routed K

Global K4 may be unnecessarily destructive. Freeze per-layer K in {4,5,6}, or
keep the smallest K whose cumulative normalized route mass exceeds a frozen
threshold. The graph may retain maximum scratch for six routes while each layer
captures its own static loop count. Dynamic compaction is a later optimization.

### H3 — exact-reranked native lm_head

Use native W4A4 only to produce a full-vocabulary shortlist, then rerun the
original weight-only lm_head exactly on the shortlisted rows with the original
activation. This can preserve the exact greedy token if held-out recall is
100%. The likely ceiling is sub-millisecond to about one millisecond, not the
remaining seven milliseconds.

### H4 — downflow rebuild

MoE remains the large block. The captured-graph marginals attribute about:

- 3.85 ms to gather;
- 2.25 ms to routed up;
- 1.81 ms to the shared expert;
- 1.37 ms to masked down;
- 1.12 ms to scan/reduce/accumulate.

Large progress requires grouped execution and fewer bytes, not another small
launch fusion.

### H5 — structural compiled derivative

If selective quantization, adaptive K, exact-reranked lm_head and downflow still
leave >3 ms, S100-single requires a compiled derivative: hybrid sublayer
pruning, expert merging/distillation, or state-width reduction with recovery.
It must be named as a derivative rather than the original checkpoint.

## Aggregate route

The E2 evidence remains stronger than the single-stream route. Q/O M16 costs
almost M1 walltime, and the measured N=16 route union averages about 64 experts
instead of 96 independent selections. One true fixed-N runtime with grouped MoE
is the most credible route to 100 aggregate tok/s.
