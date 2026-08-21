# Ornith-1.5 NVFP4/DFlash transfer report — Phase46 through Phase66

## Outcome

The Nemotron host/GPU architecture transfers to both Official Ornith-1.5
NVFP4 and Pottokao Abliterated NVFP4-DFlash, but the expert math must use
Qwen3.5 SwiGLU kernels. The checkpoint format, real weights, complete routed
expert, route-adaptive batching and cold mapped-host miss path are all measured
green on the 8 GiB RTX PRO 2000 Blackwell laptop. The exact Pottokao GGUF
target/DFlash pair also runs end to end in upstream llama.cpp and accelerates
this hybrid placement.

The new custom H4 component stack keeps 65 tok/s physically open. Its measured
all-hot floor is 51.382 ms/H4 (77.85 equivalent tok/s), leaving 10.157 ms under
the 61.538 ms/H4 boundary for routers, recurrent/full-attention cores, norms,
reductions and orchestration. This is a measured-component budget, not an
end-to-end throughput result. Cache residency is decisive: four unique misses
per layer already push the known floor to 66.558 ms/H4 before those remaining
components execute.

Upstream llama.cpp build 10549 is not lossless for this quantized draft-model
path: output is deterministic but can differ from target-only greedy decoding.
Physical target `ubatch` geometry and persistent cross-request drafter state
both affect the result. This strengthens the case for the custom runtime, where
verification geometry and drafter reset semantics can be made explicit.

## Checkpoints

| Property | Official Ornith-1.5 NVFP4 | Pottokao Abliterated NVFP4-DFlash |
|---|---:|---:|
| Text layers | 40 | 40 |
| Hidden size | 2048 | 2048 |
| Routed experts/layer | 256 | 256 |
| Experts/token | 8 | 8 |
| Routed/shared width | 512 / 512 | 512 / 512 |
| Attention schedule | 30 linear + 10 full | 30 linear + 10 full |
| Routed expert format | NVFP4 group 16 | NVFP4 group 16 |
| Text-only resident shell | 2.5197 GiB | 2.5197 GiB |
| Extra modules | vision 0.8318 GiB; MTP 1.5733 GiB | none |
| DFlash body | separate | embedded, 0.7188 GiB |

The text-only runtime deliberately omits Official's vision tower and MTP head.
That makes its resident text shell identical to Pottokao's.

## Measured evidence

### Phase46 — representation and native Blackwell execution

- Both repositories pass all layout gates.
- A real Pottokao layer-20 routed gate, routed down, shared gate and lm-head run
  through the SM120 native NVFP4 API for M1 and M8.
- M8/M1 latency ratios are approximately 0.94-1.12 across the full local
  recheck; native matrix geometry therefore amortizes eight rows almost for
  free.
- Full Pottokao snapshot and selected Official real-weight shard are local.

### Phase47 — real DFlash body

- Embedded checkpoint: 385,906,176 BF16 parameters, 771,812,352 payload bytes.
- Six-layer body is finite and bitwise deterministic.
- K8: 9.449 ms median; K16: 9.140 ms median.
- Peak allocated device memory: 792,021,504 bytes.
- These timings use synthetic target residuals/embeddings and do not measure
  acceptance.

### Phase48 — complete real SwiGLU expert

- Real layer-20 expert 0: gate + up + SiLU multiplication + down.
- Complete record: 1.6875 MiB.
- H8 versus independent byte decode: normalized RMSE 3.66e-7.
- H8 versus eight independent H1 evaluations: bit exact.
- Hot H8: 0.146 ms; eight sequential H1: 0.326 ms.
- The preregistered 4x speedup gate correctly failed: observed 2.23x. A padded
  M8 only broke even at route multiplicity 4.

### Phase49 — route-adaptive M2 through M8

Exact-size kernels eliminate padded rows:

| Multiplicity | Pottokao candidate | Speedup over sequential H1 |
|---:|---:|---:|
| M2 | 0.0558 ms | 1.46x |
| M3 | 0.0705 ms | 1.69x |
| M4 | 0.0861 ms | 1.83x |
| M5 | 0.1016 ms | 2.16x |
| M6 | 0.1158 ms | 2.31x |
| M7 | 0.1333 ms | 2.27x |
| M8 | 0.1466 ms | 2.60x |

All outputs are deterministic and approximately 3.6e-7 NRMSE from the
independent decode. The first beneficial grouped multiplicity is now 2.

### Phase50 — real Official checkpoint parity

The same M2-M8 family passes every gate on the actual Official layer-20 expert.
Official/Pottokao candidate latency ratios are 0.987-1.006. This establishes
shape and kernel portability; the abliteration does not alter the runtime path.

### Phase52 — cold mapped-host cache misses

Phase51's one-record warm-UVA result was rechecked with 80 rotating pinned
records: a 135 MiB working set, greater than four times the measured 32 MiB L2.

| Multiplicity | Hot device | Full H2D stage | Direct mapped host | Direct vs stage |
|---:|---:|---:|---:|---:|
| M1 | 0.0453 ms | 0.1993 ms | 0.1165 ms | 0.584x |
| M2 | 0.0569 ms | 0.2277 ms | 0.1392 ms | 0.611x |
| M4 | 0.0862 ms | 0.2432 ms | 0.1661 ms | 0.683x |
| M8 | 0.1466 ms | 0.3068 ms | 0.2236 ms | 0.729x |

The runtime decision is therefore:

1. Keep popular complete experts resident in the GPU cache.
2. Dispatch cache hits through H1 or exact M2-M8 according to route
   multiplicity.
3. Execute a single unique miss directly from a bounded mapped-pinned ring.
   Phase62 supersedes this rule for bulk misses: stage two or more misses unless
   a later exact count-specific measurement promotes direct-UVA.
4. Prefetch pageable checkpoint bytes into that bounded ring; do not attempt to
   pin the full 16.875 GiB routed payload.

### Phase53 — independent llama.cpp target + DFlash

- Exact Pottokao target and 782.8 MB DFlash GGUFs, official CUDA 13.3 build
  10549, 10 fixed target GPU layers and all draft layers on GPU.
- Geometric-mean wall throughput: 9.93 to 11.44 tok/s, a 1.153x speedup.
- 99/198 drafted tokens accepted (0.500).
- Coding was byte-exact; arithmetic diverged, so the preregistered correctness
  gate failed despite the speed gate passing.

### Phase54 — fresh-process K=1/K=8 diagnosis

- Baseline, K=1 and K=8 were each byte-deterministic across two fresh-process
  replicates with prompt caching disabled.
- K=1 accepted 30/33 and K=8 accepted 51/96, but both speculative paths
  deterministically differed from the stable target-only arithmetic output.
- The target-only margin between `Find` and `Calculate` at the first K=1 fork
  was only about 0.196 log-probability units.
- Divergence at K=1 rules out a K=8-only draft-quality explanation and matches
  the known llama.cpp quantized-target speculative-path failure mode.

### Phase55 — target physical-ubatch geometry

Target-only was valid at `ubatch` 4 through 512. DFlash K=8 required at least
16. Against the target-only output at the same geometry, speculative output was
byte-exact at 16, 64 and 256, but not at 32, 128 or 512.

| Target ubatch | Same-geometry exact | Baseline tok/s | DFlash tok/s | Speedup |
|---:|:---:|---:|---:|---:|
| 16 | yes | 10.147 | 9.667 | 0.953x |
| 32 | no | 8.561 | 9.078 | 1.060x |
| 64 | yes | 8.639 | 8.664 | 1.003x |
| 128 | no | 8.477 | 8.389 | 0.990x |
| 256 | yes | 8.374 | 8.743 | 1.044x |
| 512 | no | 8.340 | 9.682 | 1.161x |

This is direct evidence that quantized target kernel geometry, not only accepted
draft identity, changes greedy decisions. `ubatch=256` was the best exact
single-request operating point in this sweep.

### Phase56/57 — persistent drafter state

- Repeating the two-prompt Phase53 order at `ubatch=256` gave a 1.070x median
  speedup. Coding remained exact, but the following arithmetic request diverged
  deterministically in both fresh-process replicates.
- A documented slot erase succeeded twice and removed 107 target tokens each
  time, yet post-erase arithmetic still diverged and aggregate acceptance stayed
  99/198.
- Therefore `--no-cache-prompt` and `/slots/0?action=erase` do not clear the
  persistent DFlash state used by this llama.cpp path. Strict byte-lossless
  serving currently requires a fresh worker per request or an upstream/custom
  drafter-state fix.

### Phase58 — direct-L2 FP8 H4 attention projections

- A direct-L2 FP8 E4M3 M4 kernel is bit-exact versus four M1 launches on every
  tested real Official and Pottokao projection.
- Median speedup is 2.028x; the kernel uses 28 registers and no local memory.
- Official/Pottokao latency ratios are 0.987-1.014 for the shared linear
  geometries.
- The measured 30-linear plus 10-full projection total is 23.924 ms/H4.

### Phase59 — 32-way routed-expert bulk H4

- Thirty-two real unique layer-20 expert assignments execute in 0.561 ms versus
  1.486 ms serially: 2.649x faster.
- Candidate and same-kernel serial outputs are bit-identical; independent
  dequantized-reference NRMSE is 4.25e-7.
- Projected over 40 layers, worst-case-unique hot routed work is 22.438 ms/H4.

### Phase60/61 — indirect route reuse and occupancy

- Cache-indirect M1-M4 buckets are exact and faster for repeated routes, but
  Phase60 narrowly missed its frozen M4 1.15x gate at 1.143x.
- Two warps per row reduced M4 from 64 to 56 registers and beat the matched
  one-warp M4 by 1.096x, but still missed the assignment-control gate at 1.102x.
- These failed gates close aggressive route-reuse claims. The production plan
  retains multiplicity bucketing, with inter-expert bulk parallelism as the
  primary speed mechanism.

### Phase62 — cold bulk miss crossover

All direct and staged arms are bit-exact against hot execution. Rotating working
sets are 4.06-6.75x the 32 MiB L2.

| Unique misses | Hot | Direct UVA | Bulk stage | Selected |
|---:|---:|---:|---:|---|
| 1 | 0.042 ms | 0.136 ms | 0.199 ms | direct |
| 4 | 0.088 ms | 0.671 ms | 0.449 ms | stage |
| 8 | 0.152 ms | 1.343 ms | 0.818 ms | stage |
| 16 | 0.278 ms | 2.706 ms | 1.565 ms | stage |
| 32 | 0.558 ms | 5.767 ms | 3.224 ms | stage |

The earlier one-record “always direct” conclusion therefore does not scale to
an H4 miss union. Runtime policy is `1 -> direct_uva`, `>=2 -> bulk_stage`
conservatively until counts two and three are measured.

### Phase63/64 — hybrid native/exact LM head

- Direct ERVF H1x4 costs 7.561 ms on the 248,320-token head; the best direct M4
  is only 1.54x faster, so Phase63 failed its 2.5x gate.
- Native NVFP4 activation/head top-1 alone matches only 26/32 synthetic rows.
- Native top-64 recall is 32/32. An indexed ERVF rerank reproduces every
  shortlisted full-ERVF logit bit-for-bit and restores all 32/32 token IDs.
- Complete quantize + native head + top-64 + exact rerank is 1.575 ms/H4,
  4.801x faster than ERVF H1x4. Real final-normalized Ornith activations still
  require end-to-end adjudication.

### Phase65 — shared-expert overlap closed

Shared/routed outputs remain bit-exact, but overlap is 0.698 ms versus 0.694 ms
serial. Both branches compete for memory bandwidth, so Nemotron's shared-stream
overlap does not transfer and is excluded from the budget.

### Phase66 — 65 tok/s budget

| Known all-hot component | H4 latency |
|---|---:|
| FP8 attention projections | 23.924 ms |
| Routed experts, 40 layers | 22.438 ms |
| Shared experts, 40 layers | 3.444 ms |
| Native top-64 + exact head | 1.575 ms |
| **Known floor** | **51.382 ms** |
| **Residual to 65 tok/s** | **10.157 ms** |

One unique miss in every layer raises the known floor to 56.148 ms/H4; four in
every layer raise it to 66.558 ms/H4. If 5 ms is reserved for other unmeasured
work, no more than roughly 43 isolated misses across 1,280 H4 assignments fit,
which corresponds to a minimum 96.6% hit rate. A real route trace is required
to determine whether the 52-expert/layer cache reaches that boundary.

### Phase67 — fused Qwen3.5 linear-attention H4

The full post-projection Gated DeltaNet core now executes in three H4 launches:
fused A/B gates, depthwise causal convolution, and recurrent delta update plus
gated RMSNorm. The benchmark uses real layer-20 auxiliary weights and the full
2 MiB FP32 recurrent state.

| Checkpoint | One linear layer | 30 layers | Output NRMSE | State NRMSE |
|---|---:|---:|---:|---:|
| Official Ornith-1.5 | 0.0652 ms | 1.955 ms | 2.72e-7 | 1.61e-7 |
| Pottokao abliterated | 0.0655 ms | 1.965 ms | 2.72e-7 | 1.31e-7 |

All kernels use zero local memory; the recurrence uses 38 registers. Fresh
states reproduce bit-identical GPU outputs. Adding the worse 30-layer result
to the Phase66 floor gives 53.347 ms/H4, or 74.98 tok/s equivalent, and leaves
8.192 ms/H4 for full attention, routers, remaining norms and orchestration.

### Phase68 — full causal-attention H4

The complete post-projection core includes Q/K RMSNorm, partial RoPE, four K/V
appends, exact intra-H4 causality, softmax attention and the query output gate.
Three predeclared CTA geometries shared one, four or all eight Q heads of a KV
group. G1 won at both contexts: at these sizes the extra grid parallelism is
worth more than eliminating repeated K/V reads.

| Context | Selected | Official | Pottokao | Worse ten-layer cost |
|---:|---|---:|---:|---:|
| 128 | G1 | 0.0582 ms/layer | 0.0579 ms/layer | 0.582 ms/H4 |
| 1024 | G1 | 0.3500 ms/layer | 0.3503 ms/layer | 3.503 ms/H4 |

All arm outputs have NRMSE below 7.1e-7, are deterministic, and use zero local
memory. The conservative ctx1024 known floor is now 56.849 ms/H4 (70.36 tok/s
equivalent), leaving 4.689 ms/H4 to the 65 tok/s boundary. At ctx128 the full
attention contribution is only about 0.582 ms/H4.

### Phase69 — routers, norms and reductions

The remaining all-hot support path now includes 81 H4 RMSNorm reductions,
40 real 256x2048 router projections, 40 shared-gate projections, device top-8
selection/normalization, all-hot slot lookup, route-order expert reduction,
shared gating and both residual transitions. Residual addition and the next
norm are fused at each transition.

| Checkpoint weights | Complete 40-layer support |
|---|---:|
| Official | 1.925 ms/H4 |
| Pottokao | 1.757 ms/H4 |

Top-8 IDs, slots and hit flags are exact; maximum numerical NRMSE is 4.64e-7.
All kernels use zero local memory. The budget also adds a conservative 1.321 ms
for Phase60 cache-indirect M1 versus Phase59 contiguous bulk32, even though the
absolute measurements came from separate sessions.

The resulting conservative ctx1024 known floor is 60.095 ms/H4, equivalent to
66.56 tok/s, with 1.443 ms remaining to 65 tok/s. At ctx128, substituting its
0.582 ms ten-layer attention cost gives approximately 57.175 ms/H4, or 69.96
tok/s. These remain component floors, not an end-to-end throughput claim.

### Phase70 — real routes and final activations

An unmodified llama.cpp eval callback now captures every real
`ffn_moe_topk-*`, normalized route weight and `result_norm` tensor. The custom
runner marks all four H4 tokens as outputs, avoiding the normal last-row-only
LM-head graph. Two fresh CPU callback runs are exactly equal; CPU is used only
for observability, not for latency claims.

- All 40 route tensors have shape 8x4, all 40 weight tensors normalize within
  6.0e-8 of one, and all values are finite.
- Routes, route weights and the complete 2048x4 final activation are exactly
  reproducible across fresh contexts.
- With the real final activations, the native top-64 contains all exact top-32
  logits for every token: 128/128 recall. Exact ERVF rerank restores the full
  top-32 order 128/128 and all shortlisted scores are bit-exact.
- The prior 1.575 ms/H4 head path is therefore no longer synthetic-only.

A fixed 128-token trace identifies expert residency as the remaining barrier:

| 52 slots/layer | Assignment hit rate | Warm unique misses/layer/H4 | Warm unique misses/all layers/H4 |
|---|---:|---:|---:|
| LRU | 72.37% | 8.05 | 321.82 |
| Belady oracle | 82.66% | 4.30 | 172.18 |

Belady is an unattainable future-aware upper bound, yet it still leaves only
1.82 zero-miss layers per H4 on average. Thus Phase69's 66.56 tok/s result is a
valid all-hot component floor, but plain 52-slot replacement cannot turn it
into a 65 tok/s end-to-end result. The next experiment must hide or eliminate
real miss transport; another hot-kernel micro-optimization cannot close this
gap.

### Phase71 — real-trace prefetch oracle

The exact Phase70 miss schedule was converted to real pinned-H2D bytes and
overlapped on a dedicated CUDA stream with the measured 60.095 ms/H4 compute
envelope. The envelope is an optimistic one-SM wait proxy, so this is a ceiling
for copy-engine scheduling rather than an end-to-end result.

| Policy | Serial | Best reset-per-H4 prefetch | Exposed tail | Equivalent |
|---|---:|---:|---:|---:|
| LRU-52 | 90.650 ms | 61.951 ms (lead 4) | 1.992 ms | 64.57 tok/s |
| Belady-52 | 80.149 ms | 61.641 ms (lead 2) | 1.682 ms | 64.89 tok/s |

The source working set is 6.54x L2 and the calibrated waits are within 0.3% of
their targets. Both the implementable and oracle policies miss the frozen
1.443 ms exposed-tail gate. However, the schedule restarts its pipeline at
every H4 and repeatedly exposes layer-0 transfer. A continuous cross-H4 ring
is the final prefetch-only test justified by this result; it can move the next
block's initial copy under the current block's late compute/head.

### Phase72 — cross-H4 rolling prefetch breakthrough

The same real miss bytes and 52-slot policies were scheduled as one continuous
28x40-layer ring. Copies for the next H4 may now run under the current H4's
late layers and final head instead of restarting the pipeline.

| Policy | Selected rolling lead | Exposed tail | Floor-normalized H4 | Equivalent |
|---|---:|---:|---:|---:|
| LRU-52 | 4 | 0.500 ms | 60.595 ms | 66.01 tok/s |
| Belady-52 | 2 | 0.473 ms | 60.569 ms | 66.04 tok/s |

All preregistered gates pass. The LRU/Belady tail gap is only 0.027 ms, showing
that after continuous copy scheduling, ordinary LRU replacement is no longer
the oracle's dominant disadvantage on this trace. This is the first real-route
Ornith configuration whose measured component floor plus real miss-byte DMA
fits the 65 tok/s envelope. The claim remains an optimistic DMA oracle because
expert records are copied contiguously and the compute envelope is represented
by a non-VRAM-contending calibrated wait kernel. Segmented copies and real
kernel integration are required before an end-to-end claim.

## Transfer matrix

| Existing research component | Ornith status | Reason |
|---|---|---|
| NVFP4 byte loader/layout | Transfers | Same E2M1 + E4M3 + F32 group-16 contract |
| Native SM120 matrix path | Transfers | Real target and head tensors measured green |
| Phase33 H8 idea | Transfers with M2-M8 family | Exact-size dispatch avoids padded work |
| GPU expert cache | Transfers | 3.4278 GiB budget holds 52 complete experts per layer across 40 layers |
| Mapped-host miss path | Transfers conditionally | Direct wins for one miss; bulk staging wins from four measured misses onward |
| Prefetch/cache-policy research | Transfers conceptually | Requires real Ornith route trace |
| FP8 direct-L2 H4 projections | Transfers | Real Official/Pottokao M4 is exact and about 2x faster |
| Inter-expert bulk dispatch | New Ornith path | 32 unique routed assignments are exact and 2.65x faster |
| Shared-expert stream overlap | Does not transfer | Memory-bandwidth contention erases the overlap |
| Native-head acceleration | Transfers with exact rerank | Native top-64 plus indexed ERVF recovers exact selected IDs on 32 synthetic rows |
| Qwen3.5 linear-attention H4 | Transfers | Full conv/gate/delta/norm core is independently green at 1.965 ms over 30 layers |
| Qwen3.5 full-attention H4 | Transfers with G1 dispatch | Q/K norm, RoPE, causal cache, attention and output gate are green at ctx128/1024 |
| Routers/norms/residual reductions | Transfers with fusion | Full 40-layer support costs at most 1.925 ms/H4 and route IDs are exact |
| Real final-activation head path | Transfers | Native top-64 retains exact top-32 128/128; rerank order and scores are exact |
| 52-slot LRU residency | Insufficient alone | Real warm trace leaves 321.82 unique layer/expert misses per H4 |
| Nemotron ReLU2 sparse down | Does not transfer | SwiGLU output is dense; no exact-zero column mask |
| Nemotron Mamba/SSM kernels | Replaced | Qwen3.5 recurrence now has a dedicated fused H4 implementation |
| Nemotron hardcoded DFlash adapter | Does not transfer | Different target layers, hidden injection and draft semantics |

## Remaining critical path

1. Replay the real Phase70 miss schedule through copy-engine double buffering
   and layer-ahead DFlash prefetch. Measure the exposed tail after overlap, not
   the sum of isolated copy times. Belady supplies the optimistic lower bound;
   LRU is the implementable baseline.
2. If exposed transport exceeds 1.443 ms/H4, test a larger effective resident
   set through compressed cold-expert tiers or a reduced non-expert GPU
   footprint; do not reinterpret the all-hot component floor as end-to-end.
3. Add explicit per-request target/drafter state reset and fixed verification
   geometry to the custom runtime; gate every speculative completion against
   target-only greedy output during bring-up.

The Official 1.5 repository currently provides safetensors rather than an
Official 1.5 GGUF. Phase50 therefore establishes real-weight Official kernel
parity, while the independent llama.cpp generation phases necessarily use the
published Pottokao target/DFlash GGUF pair.
