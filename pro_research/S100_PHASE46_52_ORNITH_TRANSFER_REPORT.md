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
| Nemotron ReLU2 sparse down | Does not transfer | SwiGLU output is dense; no exact-zero column mask |
| Nemotron Mamba/SSM kernels | Replaced | Qwen3.5 recurrence now has a dedicated fused H4 implementation |
| Nemotron hardcoded DFlash adapter | Does not transfer | Different target layers, hidden injection and draft semantics |

## Remaining critical path

1. Port and independently gate Ornith's 40 routers plus remaining layer norms,
   residual reductions and graph orchestration. At ctx1024 they must fit inside
   the remaining measured 4.689 ms all-hot residual.
2. Capture real target router IDs for H4 speculative blocks and replay them
   through the implemented multiplicity planner, 52-expert/layer cache and
   miss-count-adaptive transport. With 5 ms reserved elsewhere, approximately
   96.6% route hits are required.
3. Validate native top-64 recall on real final-normalized Ornith activations;
   increase shortlist size or fall back to full ERVF whenever the exact winner
   is not provably retained.
4. Add explicit per-request target/drafter state reset and fixed verification
   geometry to the custom runtime; gate every speculative completion against
   target-only greedy output during bring-up.

The Official 1.5 repository currently provides safetensors rather than an
Official 1.5 GGUF. Phase50 therefore establishes real-weight Official kernel
parity, while the independent llama.cpp generation phases necessarily use the
published Pottokao target/DFlash GGUF pair.
