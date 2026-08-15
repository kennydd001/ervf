# PRO research — second breakthrough pass

Date prepared: 2026-08-15  
Base commit: `96811c4e381bf788f9133f5d1fc025e6885cf78f`  
Target: Nemotron 3.5 Lightning 30B-A3B NVFP4, batch 1, RTX PRO 2000 Blackwell Laptop 8 GiB

## Why this pack exists

The newest Kimi/Claude work materially changed the state of the project:

- device-resident routing + device LRU is now built and measured: about
  `41.540 -> 36.998 ms/token` in its registered eager A/B;
- the complete full-token CUDA graph is now **implemented**, but the required
  EGR/GRAPH/CTL/DET runner and verifier were still missing;
- ERVF is proven exact and large on NVFP4, but the resident BF16, FP8-tensor and
  FP32 GEMVs still use the pre-ERVF one-row/256-thread reduction geometry;
- the graph implementation is causal: the graph's argmax writes the next token
  to the device token buffer consumed by the next replay.

Therefore the correct next move is not another broad idea campaign. It is to
measure the built graph and test two tightly derived generalisations of the one
mechanism that already worked.

No file in the closed research namespaces is modified by this pack. Every run
writes only to `pro_research/results/` and archives an older result before
writing a replacement.

## G0 — execute E1F22 exactly as preregistered

### Hypothesis

Once routing, LRU assignment, miss staging, position, argmax and token state are
on-device, capturing the full token should recover at least 2.5 ms beyond the
already measured eager device-cache path.

### Frozen gates

- GRAPH IDs equal EGR IDs for 3 x 256 generated tokens.
- First 64 IDs of each two anchor prompts equal the frozen A1 IDs.
- A deliberately sabotaged `bad_pick=1` graph diverges on at least one prompt.
- Repeated graph replay is deterministic.
- `p50(GRAPH) <= p50(EGR) - 2.5 ms` over at least 500 timed tokens.
- Graph/device-buffer VRAM growth is under 64 MiB.

### Why this is first

The implementation already exists. Until it is measured, any new runtime
architecture is premature. A pass would turn a theoretical 8.9 ms graph oracle
into a causal physical runtime result; a fail identifies the exact remaining
launch/capture problem.

## G1 — generalized ERVF for the resident shell

### Observation

NERVF accelerated the critical NVFP4 GEMV by 1.936x by remapping the exact
logical reduction tree onto 16-lane subwarps. Yet the current runtime still uses
one full 256-thread block per output row for:

- BF16 Q/K/V/O attention projections;
- FP8-per-tensor Mamba input/output projections;
- FP32 MoE router matrices.

These kernels contain the same two-level 256-thread reduction structure that
ERVF was designed to virtualize.

### Hypothesis

The exact same mapping can process 16 rows per physical block while preserving:

- every virtual thread's MAC sequence;
- every FMA;
- the original offset-16, 8, 4, 2, 1 reduction tree;
- the final FP32 bits.

### Fail-closed gates

Microbench opens integration only when:

- every real checkpoint shape is bit-identical;
- geometric-mean speedup is at least 1.25x;
- no registered shape is more than 5% slower.

Full-model pass additionally requires:

- BASE-A, candidate and BASE-B token sequences all identical;
- candidate p50 improves by at least 1.5 ms or 5%;
- BASE-A/BASE-B p50 drift is at most 1 ms.

### Why this can be larger than another small kernel tweak

It touches many resident projections in all 52 layers, including the 23 Mamba
layers and 23 routers, rather than only the routed-up NVFP4 projection.
Whether the speed survives end-to-end is unknown; the A/B/A gate prevents a
microbenchmark from being promoted into a token-level claim.

## G2 — exact K-token epoch graph

### Observation

The full-token graph is already autoregressively causal: argmax writes the next
ID into device memory. The host does not need to inspect that ID before the next
graph replay unless it wants to stream every token immediately.

### Hypothesis

If CUDA permits the existing token graph to be embedded as child nodes inside a
parent graph, one parent launch can advance K exact tokens. That removes K-1
host graph submissions and amortizes readback/synchronization without a draft
model or speculative acceptance.

### Gates

For K in `{2,4,8,16,32}`:

- parent-graph IDs exactly equal K separately queued child replays;
- p50 time per token is at least 1.10x faster;
- extra VRAM is below 64 MiB.

Unsupported nested capture is a valid technical closure. The runner does not
silently substitute another algorithm.

### Claim boundary

This is an offline/queued single-stream throughput method. Interactive latency
still depends on how often IDs are harvested. It is not multiple-token
prediction and does not change target semantics.

## What this pack deliberately does not do

- It does not reopen gatherless downflow; that exact path was slower because of
  strided PCIe reads.
- It does not reopen low-rank ReLU2 prediction, RSIV or GhostWeights.
- It does not add speculative decoding after the existing negative target
  verifier evidence.
- It does not claim TMA direct-to-SMEM support on this Windows/CuPy stack before
  a compilable hardware probe exists.
- It does not add component speedups together.

## Decision table after manual runs

| Result | Meaning | Next action |
|---|---|---|
| G0 passes | causal graph-resident token is real | adopt on a separate phase, then rerun G1 inside graph |
| G0 fails correctness | graph implementation bug/semantic issue | fix only the identified failure; do not tune speed |
| G0 passes correctness but not speed | graph replay is not the missing 2.5 ms | profile graph nodes; keep device-cache eager baseline |
| G1 micro passes and integration passes | ERVF generalizes beyond NVFP4 | combine G0+G1 in a new physical A/B |
| G1 micro fails exactness | generalized mapping/compiler differs | close exact track; no tolerance widening |
| G2 passes | K-token causal graph epochs are a new large lever | integrate best K with G0/G1 and measure streaming trade-off |
| all fail | current exact architecture likely needs a new memory/runtime primitive | only then open a TMA/DAK or llama.cpp-fork branch |

## Breakthrough threshold

This pack does not define success as “one more fast kernel.” A product-level
breakthrough requires an independently verified integrated causal run of at
least:

- `>= 50 tok/s` short-context exact decode, or
- a comparably strong long-context result,

with p50/p95/p99, VRAM, long rollout and thermal behavior reported. The scripts
will report intermediate gates honestly even when this threshold is missed.
