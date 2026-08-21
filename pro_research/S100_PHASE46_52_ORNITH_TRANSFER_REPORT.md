# Ornith-1.5 NVFP4/DFlash transfer report — Phase46 through Phase57

## Outcome

The Nemotron host/GPU architecture transfers to both Official Ornith-1.5
NVFP4 and Pottokao Abliterated NVFP4-DFlash, but the expert math must use
Qwen3.5 SwiGLU kernels. The checkpoint format, real weights, complete routed
expert, route-adaptive batching and cold mapped-host miss path are all measured
green on the 8 GiB RTX PRO 2000 Blackwell laptop. The exact Pottokao GGUF
target/DFlash pair also runs end to end in upstream llama.cpp and accelerates
this hybrid placement.

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
3. Execute misses directly from a bounded mapped-pinned ring instead of first
   staging the complete record to a device mirror.
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

## Transfer matrix

| Existing research component | Ornith status | Reason |
|---|---|---|
| NVFP4 byte loader/layout | Transfers | Same E2M1 + E4M3 + F32 group-16 contract |
| Native SM120 matrix path | Transfers | Real target and head tensors measured green |
| Phase33 H8 idea | Transfers with M2-M8 family | Exact-size dispatch avoids padded work |
| GPU expert cache | Transfers | 3.4278 GiB budget holds 52 complete experts per layer across 40 layers |
| Mapped-host miss path | Transfers | Cold rotating direct-UVA wins 27-42% over staging |
| Prefetch/cache-policy research | Transfers conceptually | Requires real Ornith route trace |
| Nemotron ReLU2 sparse down | Does not transfer | SwiGLU output is dense; no exact-zero column mask |
| Nemotron Mamba/SSM kernels | Does not transfer directly | Qwen3.5 uses a different linear-attention recurrence |
| Nemotron hardcoded DFlash adapter | Does not transfer | Different target layers, hidden injection and draft semantics |

## Remaining critical path

1. Add explicit per-request target/drafter state reset and fixed verification
   geometry to the custom runtime; gate every speculative completion against
   target-only greedy output during bring-up.
2. Capture real target router IDs for speculative blocks and build the M1-M8
   multiplicity histogram per layer.
3. Replay that trace through the 52-expert/layer cache plus bounded pinned-ring
   miss path.
4. Port Qwen3.5 linear attention and full attention into the custom runtime,
   gated against an independent reference before reporting custom end-to-end
   tok/s.

The Official 1.5 repository currently provides safetensors rather than an
Official 1.5 GGUF. Phase50 therefore establishes real-weight Official kernel
parity, while the independent llama.cpp generation phases necessarily use the
published Pottokao target/DFlash GGUF pair.
