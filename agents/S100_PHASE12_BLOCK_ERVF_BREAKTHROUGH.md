# S100 Phase 12 — Block-ERVF speculative mini-prefill

Date: 2026-08-18
Parent evidence: Phase 9/10 close cache, offload and same-byte M=1 kernel tuning as routes to 100 tok/s.

## Central falsifiable claim

The remaining route to >100 useful single-stream tok/s is to stop paying one active-weight pass per emitted token. Convert autoregressive decode into short target-verified mini-prefills, then make the verifier reuse every weight load across the whole draft block.

This is a joint algorithm/runtime/kernel design:

1. a parallel block drafter predicts B future tokens;
2. the target verifies the block losslessly in one forward pass;
3. dense Mamba/attention/lm-head projections use multi-row ERVF (ERVF-M), preserving each row's exact reduction tree while reading each weight tile once;
4. every MoE layer transposes `(token, expert)` routes into expert groups, fetches each unique expert once, executes all assigned token rows together, and applies route weights in the down-projection epilogue;
5. Mamba and KV state are written to shadow trajectories; only the state at the accepted prefix length is committed.

## Why prior approaches failed

- Cache/miss path is too small and panel indirection costs more than it saves.
- Arc full downflow moves work to a slower engine; Arc N=1 miss economics are not enough.
- Mamba FP8 stream tuning wins 6% in isolation but only 0.012 ms/token integrated because that stream is already overlapped.
- Global W4 or top-k reduction buys latency but fails frozen fidelity.
- Earlier speculative economics charged route-union work without a true grouped token-expert verifier, so multi-token verification did not reuse expert weights.

## Architecture

### Parallel drafter

Train and compare three frozen draft families using target-generated traces:

- FastMTP-style recursive shared MTP head, B in {2,4,8};
- PARD-style one-pass parallel draft;
- DFlash-style target-hidden-conditioned block drafter.

Nano has no native MTP checkpoint, so the target remains frozen and only the drafter is trained. Greedy verification must produce exactly the target's greedy sequence. A suffix-automaton fallback may override neural drafts only on exact prompt/history matches.

### ERVF-M

For B activation rows, each kernel:

- reads a weight tile once;
- maintains B independent virtual accumulator sets;
- reproduces the current ERVF reduction tree independently for every row;
- supports FP8 Mamba in/out, BF16/FP8 attention projections, NVFP4 QFAST shapes, shared expert and lm_head;
- measures B={1,2,4,8,16} under >=4x-L2 weight rotation.

The hard gate is exact output plus useful-row throughput >=1.75x at B=2, >=3.2x at B=4 and >=5.5x at B=8 versus B independent M=1 calls.

### Block Mamba

All B input token ids are known during verification. Batch the dense projections with ERVF-M. Run the selective state recurrence over B steps inside one layer kernel or associative scan, starting from the canonical state and writing a state trajectory. Do not mutate canonical state until acceptance length is known.

### Token-expert transpose

For each MoE layer:

1. route all B positions;
2. stable-sort the B*6 `(expert, token, route_weight)` records by expert;
3. deduplicate the expert union;
4. fetch/cache each unique expert once;
5. run grouped routed-up over all rows assigned to each expert;
6. fused ReLU2;
7. grouped routed-down with route-weighted scatter/reduction.

Use SM120 native narrow-precision CUTLASS/CuTe kernels or a custom grouped ERVF path. Per-expert discrete pointers are required because weights are not one contiguous VRAM tensor. Correctness, nonuniform known values, real checkpoint samples and sabotage arms precede timing.

### Shadow commit

- Mamba: verifier writes B candidate states; commit state[accepted_prefix].
- Attention: KV tail is written speculatively; advance the canonical length only by the accepted prefix and overwrite rejected slots next round.
- MoE cache/LRU: update canonical residency only for accepted positions, or keep speculative fetches as nonsemantic prefetches while preserving deterministic routing outputs.

## Throughput gate

For a verification cycle with wall time T and accepted tokens A (including any bonus correction token):

`useful_tok_s = 1000 * A / T`.

S100 requires the lower 95% confidence bound to exceed 100 tok/s on a frozen mixed-domain prompt set, with output token identity against the current quality-green parent.

Necessary early gate:

- B=4: median accepted tokens >=2.8 and verifier <=28 ms;
- or B=8: median accepted tokens >=4.0 and verifier <=40 ms.

Draft time, rollback/commit, route grouping, expert fetch, sampling and synchronization are included.

## Decision sequence

1. Instrument a B-token exact verifier using ordinary kernels, solely to prove block state/KV correctness.
2. Measure route-union and per-expert row-count distributions for B={2,4,8} on 10k target tokens.
3. Build ERVF-M dense kernels and grouped MoE microkernels against real weights.
4. Integrate B=2, then B=4. Do not train a large drafter before target verify economics are green.
5. Train MTP/PARD/DFlash drafters only after the verifier predicts a break-even acceptance length.
6. Run end-to-end exact greedy verification, then stochastic exactness if needed.

## Kill criteria

Close this route if, after true grouped verification:

- B=4 verifier remains >35 ms, or
- the expert union gives <20% routed-weight reuse, or
- all trained drafters yield median accepted length <2.5 at B=4, or
- shadow-state commit cannot reproduce baseline tokens exactly.

If closed, 100 tok/s requires an explicitly quality-trained elastic derivative rather than the original 30B parent.

## Parallel practical route

NVIDIA's released Elastic 30B-A3B checkpoint contains trained 23B and 12B nested variants. Porting the 12B/2A variant to ERVF is the most credible quality-trained derivative path: materially fewer bytes per token, followed by the same block-verification runtime. This is a different model and must be evaluated as such.
