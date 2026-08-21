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

### Phase73 — segmented copies under real routed compute

The contiguous oracle copy was replaced by the exact six checkpoint segments
per expert: three 524,288-byte code planes and three 65,536-byte scale planes.
Every simulated layer now runs the real Pottokao bulk32 gate/up/SwiGLU/down
kernel while copies execute. Only the remaining non-routed part of the measured
layer envelope is represented by a wait kernel.

| Policy | Selected lead | Exposed tail | Floor-normalized H4 | Equivalent |
|---|---:|---:|---:|---:|
| LRU-52 | 2 | 0.158 ms | 60.253 ms | 66.39 tok/s |
| Belady-52 | 2 | 0.790 ms | 60.885 ms | 65.70 tok/s |

The real hot kernel is 0.560 ms, effectively identical to Phase59's 0.561 ms,
and output remains bit-exact and repeat-exact. All gates pass. Belady's slower
wall time despite fewer transferred bytes reveals order/thermal drift between
long epochs; a paired compute/overlap/compute bracket is required before the
smaller LRU tail is treated as stable. Both policies remain below the 65 tok/s
boundary even under that conservative observation.

### Phases74–79 — causal route prediction and real DFlash signal closed

The Phase72/73 result is an oracle because it knows future target routes. Six
causal replacements were tested against held-out real routes:

| Phase | Signal | Best physical result | Verdict |
|---|---|---:|---|
| 74 | route history / recency | 16.92% miss recall; 46.59 tok/s estimate | closed |
| 75 | current-block earlier-layer route IDs | 23.74% recall; 47.76 tok/s | closed |
| 76 | exact earlier-layer target hidden through future router | 63.25% lead-2 recall; 56.79 tok/s | closed |
| 77 | online ridge residual correction | 64.63% lead-2 recall; 57.10 tok/s | closed |
| 78 | real DFlash `result_norm` through target routers | 34.95% same-position H4-union recall | closed |
| 79 | held-out learned DFlash-to-target correction | 41.44% same-position H4-union recall | closed |

The real target+DFlash callback alignment was validated over nine speculative
events. DFlash emits fixed H8 hidden rows while target batches have lengths
8/7/4; prefix alignment is exact. Target-router parity on authoritative target
hidden is 20,480/20,480 assignments. The local target config advertises one MTP
layer, but its safetensor index contains no MTP/next-token/layer-40 tensors, so
this checkpoint cannot supply a real MTP path.

### Phases80–81 — honest same-layer transport remains too slow

Phase80 removes prediction entirely: after the authoritative router, it copies
real gate/up segments, computes resident experts, then copies/consumes down
segments. Outputs are bit-exact and repeat-exact, but the paired exposed tail is
10.133 ms/H4. Floor-normalized throughput is only **56.96 tok/s**.

Phase81 decomposes that tail over the complete 28x40 miss schedule. A thermally
invalid first run is retained in history; the mirrored rerun is stable. Eager
split dispatch adds effectively nothing (-0.080 ms/H4), while CUDA Graph split
dispatch adds 0.383 ms/H4. The Phase80 tail is therefore transport/readiness,
not Python or kernel-launch overhead. CUDA Graph work on this arm is closed.

A production-side `RollingPrefetchController` now implements persistent LRU52
metadata, temporary staging rings, exact commit, rejection/partial-acceptance
barriers, abort/reset semantics and route-order execution plans. It preserves
the oracle mechanism without pretending a causal predictor exists.

### Phase83 — 66.57 tok/s reproduces, but not at long context

Phase69 was repeated after the causal audit and remains green at **60.0842
ms/H4 = 66.5733 tok/s**. This is still an all-hot component stack, not complete
DFlash generation. Phase83 substitutes measured full-attention cost at real KV
lengths into that same stack:

| Context | Selected full-attention arm | Component floor | FP32 KV, 10 full layers |
|---:|---|---:|---:|
| 1,024 | g1 | 66.53 tok/s | 0.039 GiB |
| 4,096 | g1 | 57.17 tok/s | 0.156 GiB |
| 16,384 | g1 | 23.13 tok/s | 0.625 GiB |
| 50,000 | g1 | 11.32 tok/s | 1.908 GiB |
| 100,000 | g1 | 5.66 tok/s | 3.815 GiB |

Every context remains reference-correct, deterministic and finite. The failure
is physical: full attention scales with context and the current FP32 KV format
exceeds the frozen 0.5-GiB runtime reserve from 16k onward. The 66.57 number
must not be presented as a 50k/100k or end-to-end result.

### Phase84D — DFlash-candidate-workload MoE/transport stress test

The nine candidate-induced target batches from the instrumented target+DFlash
run were replayed through the custom BF16 router, transactional LRU52 cache,
exact six-segment NVFP4 transport, route-adaptive routed experts and shared
expert for all 40 layers. The two earlier callback groups warm each layer's
cache; partial 7- and 4-row batches use fixed H4 kernel geometry without
committing padded rows.

- Custom top-8 routes match every authoritative callback row at all 40 layers.
- Two fresh-cache executions are bit-identical and finite.
- The workload contains 67 real target rows, represented by 17 H4 launches.
- LRU52 performs 3,792 unique layer/expert copies, or 5.576 misses per
  layer/H4 and 6.710 GB of real compressed expert traffic over the trace.
- The first full sweep measured **67.376 ms/H4**. Independent three-repeat
  medians measured **66.956** and **65.693 ms/H4** for the same 40-layer
  MoE/router/cache/transport workload.

This is explicitly a **DFlash-candidate-workload MoE/transport stress test**,
not a complete verifier, not a complete decoder and not output tok/s. It proves
that this candidate-induced route workload is more expensive than the earlier
60.084 ms/H4 all-hot component floor, but it does not yet distinguish an
incomplete optimistic floor from DFlash-specific cache damage. That distinction
requires the identical executor on authoritative target-only H4 blocks.

### Phase84 — target-only discrepancy localized

The identical executor was then driven only by the committed authoritative
64-token target/reference trace. Rows 0..31 warmed LRU52 and rows 32..63 formed
eight measured H4 blocks. No DFlash hidden state, route, prefetch choice or
cache signal enters this execution.

- All 40 custom routers exactly match every authoritative target route.
- Independent uninstrumented three-repeat runs measured **74.102**, **74.787**
  and **76.156 ms/H4**.
- The target sequence has **7.263 misses/layer/H4**, versus Phase84D's 5.576.
  Layer 0 alone has **18.0 misses/H4**, versus Phase84D's 15.588.
- Instrumented wall time decomposes into 22.924 ms H2D staging, 3.011 ms
  router/top-8, 28.854 ms expert/shared/combine and 20.587 ms D2D cache
  promotion per H4. The instrumented sum is 75.376 ms/H4.

This rejects the captured-workload version of the DFlash-cache-damage
hypothesis: the ordinary target/reference sequence is both miss-heavier and
slower. The Phase69 60.084 ms/H4 result is an optimistic all-hot component
floor. Its conservative 29.118-ms hot MoE allowance excludes the measured
43.511 ms/H4 of H2D staging plus D2D promotion and also understates exact-route
compute/control. A diagnostic substitution yields roughly 105.8 ms/H4, but
that is only a component comparison, not a complete verifier.

The largest removable implementation cost is the second expert-weight copy:
promote a staged physical expert page into LRU52 by swapping its page-table
mapping with the evicted page. Copying the same 1,769,472 bytes D2D after each
H2D miss costs 20.587 ms/H4 here and is not required by model semantics.

### Phase84 — layer-0 target-verifier correctness gate

The target-only verifier now starts from real token embeddings and executes
layer 0 through input RMSNorm, statically scaled E4M3 projections, Gated
DeltaNet with real convolution/recurrent state, output projection, residual
addition and post-attention RMSNorm. Against an independent CPU dequantized
HF-ModelOpt reference, all 8,192 output values pass at **2.288e-7 NRMSE** and
**2.861e-6 max absolute error**. Fresh-state repeats are bit-identical, the
FP8 probe bytes exactly match PyTorch E4M3, and both recurrent state families
are nonzero after execution.

The same activation differs from the llama.cpp trace by 2.918% NRMSE because
that GGUF stores the three tested projection families as Q8_0 while the source
checkpoint stores E4M3 FP8. GGUF remains an authoritative token/route sequence,
but it is not an exact activation reference for the original HF-FP8 runtime.

This gate also exposes a missing correctness condition in the Phase58 M4/M1
benchmark: both comparison arms quantized the unscaled activation and then
applied `input_scale * weight_scale`. The checkpoint contract instead requires
`E4M3(x / input_scale)` before applying that product. The relative M4-vs-M1
kernel result remains informative because both arms shared the input, but the
old component floor did not include the correct static quantizer or its cost.
The 60.084-ms floor must therefore be remeasured inside the integrated path.
This checkpoint covers layer 0 only; it is not a complete verifier and makes no
output tok/s claim.

### Phase84 — integrated H4 gate exposes trajectory sensitivity

One target-only H4 now executes continuously through all 40 attention and MoE
blocks, final norm, the complete control head and native-shortlist/exact-ERVF
rerank. The exact ERVF result matches the candidate path's control top-1 on all
four rows and its top-64 contains all four control IDs. However, the frozen
independent CPU-dequant trajectory gate correctly **fails**: its first route-set
divergence occurs at layer 7, final hidden NRMSE reaches 8.032%, complete-logit
NRMSE reaches 5.850%, and only three of four final top-1 IDs agree.

A same-input localization over layers 0..7 distinguishes a wrong component
from propagation between numerically different runtimes. With the candidate
input supplied independently to each CPU component, every route set is exact;
router-logit error is about 1.2e-7 to 1.5e-7 NRMSE, routed/shared MoE branch
error is at most 5.60e-7, and attention-branch error is at most 2.66e-5. Tiny
accumulation-order differences are therefore amplified by later DeltaNet and
MoE decisions; they are not evidence of a locally wrong router, expert or
attention implementation.

The CPU-dequant trajectory is consequently retained as sensitivity evidence,
not relabeled as an authoritative GPU-runtime oracle. The published checkpoint
declares the ModelOpt quantization method, but the local Transformers loader
does not implement that method and attempts to expand the compressed checkpoint
as ordinary tensors. That route was stopped before memory exhaustion. Until an
official GPU oracle can run under the 8-GiB constraint, the defensible parity
contract is component-local same-input state/logit parity plus authoritative
trace route parity and fresh-run determinism. This does not turn the failed
end-to-end CPU comparison into a pass.

### Phase84 — strict authoritative ctx64 target verifier

After the sequence-authority and state gates were tightened, the complete
executor passes on the entire committed 64-token target/reference trace. The
final H4 measures **354.936 ms**, causes **263 real misses = 6.575/layer** and
moves **465,374,292 bytes** H2D. The synchronized validation profile attributes
102.922 ms to mmap/pinned packing plus H2D, 58.914 ms to routed experts and
35.071 ms to dense projections plus attention.

Both fresh runs reproduce every final-normalized bit, all 40 route sets, all
ERVF IDs and one SHA-256 digest over every persistent recurrent/KV-state byte.
All state is finite, 40/40 same-input router controls are exact, full-head
control top-1 is exact and the runtime cache-copy ledger observes zero D2D
promotion bytes. This is the first complete Phase84 run that may be labeled a
fully authoritative target/reference-sequence H4. It remains verifier latency,
not output tok/s.

### Phase84-T1 — pack/H2D pipeline is negative

The strict ctx64 workload compared the existing main-thread pack/compute-stream
path against one bounded CPU packing worker plus a dedicated non-blocking copy
stream. Both arms ran two fresh unprofiled trajectories and selected repeat two
as the warm host-page sample, followed by a separate validation trajectory.

| Arm | First/cold | Second/warm | Outcome |
|---|---:|---:|---|
| Baseline | 330.758 ms | **187.310 ms** | selected |
| Thread + copy stream | 383.131 ms | **218.508 ms** | reject |

The pipeline regresses the warm primary wall by **31.198 ms = 16.66%**. Both
arms reproduce identical final bits, persistent-state hash, all 40 route sets,
ERVF IDs, 263 final-H4 misses, 465,374,292 H2D bytes and zero observed D2D
promotion bytes. Therefore the extra worker, cross-stream event and scheduling
do not expose useful overlap; the existing asynchronous loop already overlaps
enough packing/submission, and the added control cost dominates. This arm is
closed. It makes no output tok/s claim.

### Phase84 — synthetic long-context target-only verifier

The complete target-only executor now runs a continuous deterministic token
sequence from embedding through all 40 target layers, real DeltaNet and
full-attention state/KV, dynamic router/top-8, physical-page LRU52, mmap to
pinned to GPU expert misses, routed and shared SwiGLU, residual/norm, native
head shortlist and exact ERVF rerank. It uses no DFlash route, hidden state,
prefetch signal or acceptance state. The first 64 IDs exactly match the
committed authoritative target/reference trace. At longer contexts the current
runner repeats the trace prompt through the checkpoint tokenizer; those suffix
tokens are target-only synthetic stress input, not an authoritative generated
target sequence. LRU promotion swaps opaque physical handles and records zero
D2D payload bytes.

At synthetic ctx1024 the primary complete wall-clock H4 measured **497.403 ms** and
**492.996 ms** in two independent invocations (0.89% apart). The latest result
passes every frozen gate: the authoritative token prefix is exact, all states
and outputs are finite, a fresh empty-state/cache repeat reproduces all 40
route sets and final-normalized bits, every same-input CPU router control is
exact, and exact ERVF top-1 matches the full control head for all four rows.
These values are integrated target-verifier latency, **not output tok/s**.

The final ctx1024 H4 has **666 real misses = 16.650/layer** and moves
**1,178,476,344 bytes = 1,769,484 bytes/miss**. This is 2.29x the miss rate of
the earlier fixed target-trace stress replay and 2.99x the DFlash-candidate
stress replay. Consequently DFlash-specific cache damage is not the source of
the discrepancy. The continuously evolved target trajectory is substantially
more hostile to LRU52 than either short captured replay.

An independently repeated validation H4 places synchronizations only to
localize that result; none of these boundaries enter the primary timing:

| Integrated ctx1024 component | Diagnostic ms/H4 |
|---|---:|
| Embedding + first norm | 0.046 |
| Dense projections + DeltaNet/full attention | 41.537 |
| Router/top-8/route readback | 4.776 |
| mmap -> pinned packing + real H2D | 263.701 |
| Routed experts | 60.190 |
| Shared experts | 8.952 |
| Combine + next norm | 1.076 |
| Native shortlist + exact ERVF head | 2.340 |
| **Profiled model-path sum** | **382.619** |

Cache packing and H2D are 68.9% of the profiled model path. The remaining
profiled model work is already 118.918 ms/H4, so the old 60.084-ms value cannot
be a complete integrated floor even under a hypothetical all-hot cache. That
floor combined isolated best-case kernels, omitted the correct persistent
quantization/dispatch trajectory and assumed a much friendlier route/cache
workload. The 67.376-ms DFlash and 74.787-ms fixed target values were useful
MoE/transport stress tests, but neither was a complete target verifier.

This localizes the large discrepancy into two separate failures of the old
budget: (1) 263.701 ms of real host packing/H2D for the integrated final route
union, and (2) an optimistic isolated-kernel compute sum that does not reproduce
the 118.918-ms synchronized integrated non-transport path. The first is the
dominant measured bottleneck; the second must be optimized only against this
complete executor, not by adding component medians.

### Phase84 — persistent target-only ctx4096 verifier

The identical executor and gates also pass after a real 4,096-token synthetic
target-only prefill. Two independent primary final-H4 invocations measure
**570.913** and **504.407 ms**, have the same **622 misses = 15.550/layer**
and move **1,100,619,048 bytes**. The complete prefill plus final block incurs
553,246 expert misses and 978,959,945,064 bytes of H2D per fresh run. Used GPU
memory is 7,157,121,024 bytes, leaving 1,389,363,200 bytes free.

| Diagnostic component | ctx1024 | ctx4096 | Delta |
|---|---:|---:|---:|
| Dense projections + DeltaNet/full attention | 41.537 | 61.764 | +20.228 ms |
| mmap -> pinned packing + real H2D | 263.701 | 255.115 | -8.586 ms |
| Routed + shared experts | 69.142 | 73.447 | +4.305 ms |
| Complete synchronized profile | 382.619 | 399.170 | +16.551 ms |
| Latest primary unsynchronized wall clock | 492.996 | 504.407 | +11.411 ms |

The expected context effect is visible directly in attention, while the final
4k route union happens to be less miss-heavy than the final 1k union. The two
code-identical 4k primary walls differ by **66.505 ms = 11.65%** despite exact
routes, misses, bytes and outputs. The warmer second invocation is consistent
with a large host-page/OS-cache effect. The latest 11.411-ms 1k-to-4k wall delta
is close to the 16.551-ms diagnostic-profile delta; neither single invocation
should be treated as a universal latency distribution. The 4k correctness
result is green within its synthetic-long-context claim boundary.

## Transfer matrix

| Existing research component | Ornith status | Reason |
|---|---|---|
| NVFP4 byte loader/layout | Transfers | Same E2M1 + E4M3 + F32 group-16 contract |
| Native SM120 matrix path | Transfers | Real target and head tensors measured green |
| Phase33 H8 idea | Transfers with M2-M8 family | Exact-size dispatch avoids padded work |
| GPU expert cache | Transfers | 3.4278 GiB budget holds 52 complete experts per layer across 40 layers |
| Mapped-host miss path | Transfers conditionally | Direct wins for one miss; bulk staging wins from four measured misses onward |
| Prefetch/cache-policy research | Transfers conceptually | Requires real Ornith route trace |
| FP8 direct-L2 H4 projections | Transfers with corrected quantizer | M4 arithmetic is exact; integrated inputs must use `E4M3(x / input_scale)` and its cost must be remeasured |
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

1. Preserve the ctx1024 integrated executor as the only performance baseline;
   do not convert it to output tok/s or substitute a component sum.
2. Split host packing, copy submission and copy-engine tail diagnostics inside
   the accepted baseline. The bounded worker/copy-stream overlap arm is closed
   negative and must not be extended without a new mechanism.
3. Separately reduce mmap-to-pinned packing/H2D cost and integrated routed-MoE
   dispatch cost, accepting a change only on complete-H4 wall time with all
   parity gates preserved.
4. For 50k/100k, replace the current FP32 full-attention KV path with a measured
   quantized KV + long-context attention kernel before further expert work. At
   present both memory and attention time fail by large margins.
5. The short-context exact expert barrier remains causal transport. Reopen it
   only with a new measured premise that reduces bytes or creates an
   authoritative overlap window; route-history, cross-layer, target-hidden and
   DFlash-hidden predictors are now closed on the captured traces.

The Official 1.5 repository currently provides safetensors rather than an
Official 1.5 GGUF. Phase50 therefore establishes real-weight Official kernel
parity, while the independent llama.cpp generation phases necessarily use the
published Pottokao target/DFlash GGUF pair.
