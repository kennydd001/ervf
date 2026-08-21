# S100 Phase81 — causal split-dispatch CUDA Graph decomposition

## Hypothesis

The 10.13 ms/H4 exposed tail in Phase80 is partly caused by splitting each
32-group expert workload into resident and miss kernel launches. Capturing the
complete split projection sequence as one CUDA Graph per miss count should
remove CPU launch overhead. This experiment deliberately removes H2D transport
so split dispatch and transport cannot be confused.

## Frozen setup

- Real Pottokao Ornith-1.5 NVFP4 layer-20 weights and the Phase59 bulk kernels.
- LRU52 miss counts from all 28 warm H4 blocks of the Phase70 real trace.
- 32 resident groups and 32 second-bank groups are already on the GPU.
- For miss count `m`, the candidate executes `32-m` resident groups followed by
  `m` second-bank groups for gate, up, SwiGLU and down.
- Baselines always execute 32 contiguous resident groups.
- Paired order: eager baseline A, eager split, graph baseline, graph split,
  eager baseline B.
- Timing is GPU-event time over the full 1,120-layer trace. No synthetic support
  waits or LM head are included; deltas are divided by 28 H4 blocks.
- CUDA Graphs are captured before timing, one for every observed miss count.

## Gates

1. Real-weight contracts pass and all 28x40 trace counts are consumed.
2. CUDA Graph split outputs are bit-exact, repeatable and finite against eager
   split outputs at the maximum observed miss count.
3. Paired eager baselines differ by no more than 5%.
4. Graph split overhead is at least 30% below eager split overhead.
5. Phase80's normalized floor with graph split overhead substituted for eager
   split overhead is at or below 4000/65 ms/H4.

Gate 5 is only a dispatch screen. It does not claim that H2D transport is hidden
or that an end-to-end decoder reaches 65 tok/s.

## Frozen rerun amendment (Phase81R)

The first execution failed Gate 3: the eager baseline moved from 871.51 to
1061.36 ms/epoch (19.6% around the midpoint), so its cross-arm subtraction is
not interpretable. Before rerunning, timing is changed to five mirrored rounds.
Each round executes `EB, ES, GB, GS, GS, GB, ES, EB` or its reverse, and uses
the mean of both occurrences of an arm. This cancels a first-order thermal
trend inside every round. The original result remains in results/history.

Gate 3 for the rerun requires the median relative difference between the two
mirrored eager-baseline observations to be at most 5%. Dispatch deltas are the
median of per-round paired differences, not differences between global medians.
