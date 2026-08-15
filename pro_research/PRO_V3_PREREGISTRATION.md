# PRO V3 preregistration — graph staging repair + selective ERVF

Date frozen: 2026-08-15, before any V3 target-hardware measurement.
Base branch state: `625ef1a1ec8188575ea41e46d0d242e54c18d3c3`.

This is a NEW experiment namespace. It does not reinterpret or weaken the G0/G1
results already observed. G0 remains failed on parity and G1 remains failed on
its original all-shapes-no-regression gate.

## Evidence motivating V3

Observed G0 signature:

- GRAPH output for each prompt is the last prompt token repeated indefinitely;
- graph replay is deterministic;
- measured smoke p50 was 31.287 ms eager vs 25.954 ms graph;
- the graph speed result is NOT valid until parity is repaired.

The runtime stages prompt token ids with an asynchronous 4-byte H2D from one
pinned host location and immediately reuses that host location for the next
prompt token. V3 tests the narrow hypothesis that the source host word can be
overwritten before CUDA has consumed the previous async copy. The repair under
test synchronizes the graph stream after each PROMPT replay only. Decode replays
(`token_id=None`) remain unchanged. Therefore this repair may affect prefill/TTFT
but must not affect measured decode latency.

Observed G1 signature:

- all seven tested dense ERVF shapes were bit exact;
- Q BF16: 2.419x; O BF16: 3.052x;
- Mamba-in FP8: 2.257x; Mamba-out FP8: 2.154x;
- K BF16: 0.771x; V BF16: 0.809x; router FP32: 0.795x;
- original G1 correctly failed because it required no shape to regress >5%.

V3 therefore defines a new selective dispatch policy BEFORE measuring it:

- ERVF only for BF16 `(4096,2688)` and `(2688,4096)`;
- ERVF only for FP8-tensor `(10304,2688)` and `(2688,4096)`;
- production kernels for every other BF16/FP8 shape and for all FP32 GEMVs.

No threshold is learned from the V3 run; the four shapes above are frozen from
the already-observed G1 microbenchmark.

## V3-G0S — safe-staging graph replication

Arms:

- EGR: device-cache eager reference;
- GRAPH_SAFE: identical captured graph, but each prompt-token `step_graph(id)` is
  synchronized before the pinned staging word is reused;
- DET: two GRAPH_SAFE rollouts from reset.

Smoke: 3 prompts x 16 generated tokens. Full: 3 prompts x 256.

Hard correctness gates:

1. current two-pass argmax kernel must select the expected low-index winner in a
   synthetic tie test;
2. GRAPH_SAFE ids == EGR ids for every generated token;
3. DET A == DET B;
4. no repeated-last-prompt-token pathology on all prompts unless EGR itself does
   the same;
5. graph extra VRAM <64 MiB.

Performance is diagnostic in smoke. In full mode the existing E1F22 speed gate
is retained unchanged: graph p50 must beat eager p50 by >=2.5 ms over >=500
samples. Prompt synchronization is explicitly excluded from decode timing.

External V36/A1 anchor parity is reported separately. A local model directory
whose identity differs from the frozen anchor does NOT get silently relabeled as
anchor-equivalent.

## V3-G1B — selective generalized ERVF

A/B/A causal arms on one loaded model and bank, with cache state rebuilt before
each arm:

- BASE_A: production dense kernels;
- SELECTIVE: only the four frozen winning shapes use DenseERVF;
- BASE_B: production kernels restored.

Hard gates:

1. BASE_A, SELECTIVE and BASE_B generated ids are identical on every prompt;
2. the four selected shapes have already demonstrated bit equality in G1; V3
   records that source result but does not substitute it for causal parity;
3. BASE_A vs BASE_B p50 drift <=1.0 ms;
4. candidate speed gate: p50 improves by >=1.5 ms OR >=5% versus the mean of
   BASE_A/BASE_B p50 values.

Smoke speed is diagnostic because sample count is small. Full mode requires at
least 500 SELECTIVE timed decode samples for a speed PASS.

## Claim boundary

Neither V3 component may be added arithmetically to another speedup. A combined
50 tok/s claim requires a later integrated physical run with both mechanisms in
the same causal runtime, independent verification, tails, and a long rollout.
