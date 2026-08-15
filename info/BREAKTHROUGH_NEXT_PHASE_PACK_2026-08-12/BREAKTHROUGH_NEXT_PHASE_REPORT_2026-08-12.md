# Breakthrough next phase — stop micro-tuning, open three distinct programs

## Verdict

The locally executable kernel-idea queue is exhausted. That is useful: the
remaining opportunities are no longer minor variants. They require one of
three qualitatively different projects:

1. **PORT80B** — the immediate industry breakthrough.
2. **ERGV Compiler** — the strongest already-supported scientific systems contribution.
3. **TierFlow-R1** — the long-horizon architecture/training moonshot.

The order matters. PORT80B has the highest impact-to-risk ratio and directly
tests whether sparse-MoE throughput is governed mainly by active parameters
rather than total parameters. ERGV converts the successful exact reduction
virtualization into a general method. TierFlow is the only branch aimed at a
new model architecture rather than post-hoc serving.

## 1. PORT80B: the next mandatory experiment

Target: `Qwen/Qwen3-Coder-Next`, officially 80B total and 3B activated,
48 layers, 512 routed experts, top-10, one shared expert, 36 Gated DeltaNet
layers, 12 full-attention layers, hidden size 2048 and native 262,144-token
context.

Official sources:

- https://huggingface.co/Qwen/Qwen3-Coder-Next
- https://github.com/QwenLM/Qwen3-Coder
- https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py

### Why this is now rational

The repaired N4B-R gate established exact synthetic execution for the official
expert shape. Its locked projections are:

| Quantity | Result |
|---|---:|
| Custom aligned Q5 bank | 46.497 GiB |
| Accounted host working set + reserve | 47.806 GiB |
| Q8 device shell | 1.801 GiB |
| Resident expert compute p95 | 8.869 ms |
| Zero-cache active-set H2D | 37.202 ms |
| Conservative dense-shell p95 | 28.077 ms |
| Ideal-overlap total p95 | 65.279 ms = 15.32 tok/s |
| Fully serial total p95 | 74.148 ms = 13.49 tok/s |

Crucially, the zero-cache projection already clears 10 tok/s. The first real
port should therefore **not depend on a sophisticated cache**. Start with
natural top-10 active-set streaming. Add caching only after the no-cache
baseline closes.

### First physical gate: decide 64 versus 96 GB RAM

N4A is analytical, not a full-size pinned-memory test. Before buying hardware,
build a 46.497-GiB synthetic bank with the exact final record size and run:

- memory-mapped bank + eight pinned staging windows;
- optional full `cudaHostRegister` attempt;
- 10,000 synthetic top-10 tokens;
- zero-cache and fixed-cache traces;
- hard-fault, commit, RSS, H2D p50/p95/p99 and thermal logging.

Gate on the current 64-GB system:

```text
no hard page faults after warmup
process commit <=58 GiB
H2D p95 <=45 ms
one-hour stability
```

Only a failure authorizes 96 GB RAM. A new CPU or NPU is not authorized by the
current evidence.

### Actual port strategy

Do not write the entire hybrid decoder from scratch first. The official GGUF
already runs in llama.cpp, and the official Transformers path exposes fused
Gated DeltaNet hooks through `causal_conv1d` and `flash-linear-attention`.

Preferred strategy:

1. Use llama.cpp or the official reference as the correctness shell.
2. Replace only the routed-expert backend with the proven Q5/ERVF dataplane.
3. Keep the shared expert resident.
4. Preserve the one authoritative official top-k call.
5. Start with zero-cache natural routing.
6. Add a cache only after exact end-to-end execution.

### Model artifact strategy

The official Q5_K_M GGUF is about 56.7 GB, split into four files. It is useful
as a quality and same-hardware baseline, but its K-quant semantics differ from
the custom STREAMQ5 format.

For the custom bank, stream-convert official BF16 shards one at a time:

```text
download one shard
verify hash
quantize tensors into the append-only Q5 bank
verify record digests
delete the source shard
continue
```

This avoids a 159-GB permanent BF16 copy.

### Required gates

#### A. Real weight quality

- Q5 experts + Q8 shell.
- Validation first; test once.
- Relative CE <=2%.
- 512-token generation without collapse.

#### B. Physical 4K decode

- zero-cache first;
- p95 <=90 ms;
- mean >=10 tok/s;
- VRAM <=8 GiB;
- process RAM <=58 GiB for the initial port and <=32 GiB only if later proven.

#### C. 32K decode

N4A projects 43.17 routed slots/layer at 32K, so capacity is not the main risk.
Measure the real Gated DeltaNet/full-attention shell and report prefill and
decode separately.

#### D. Prefill

The existing sequential prefill is not product-acceptable. Implement
expert-grouped GEMM prefill:

- group prompt tokens by expert per layer;
- transfer each used expert once per chunk;
- execute a batched Q5 GEMM;
- use the chunk Gated DeltaNet path;
- report TTFT for 128/512/2048/4096 tokens.

## 2. ERGV Compiler: the strongest scientific branch

The proven kernel insight is not merely “width 16 is good.” It is:

> Preserve the exact logical floating-point reduction graph while changing
> its physical CUDA thread topology.

Build a restricted IR:

```text
ExactReductionIR
- logical accumulators
- exact pairwise-add DAG
- cast and rounding points
- FMA policy
- virtual-lane to physical-lane mapping
```

The compiler searches:

- subwarp width;
- rows per block;
- virtual accumulators per lane;
- vector load width;
- activation staging;
- scale broadcast;
- register and occupancy budget.

Hard constraint:

```text
the ordered floating-point DAG is identical to the reference
```

Objective:

```text
minimize measured p50/p95 under the exactness constraint
```

Mandatory scientific gates:

1. Reproduce the manual P7 configuration.
2. Beat it on at least one matrix family.
3. Generalize across Q5 and Q8.
4. Reproduce on Qwen3-Coder-Next shapes.
5. Reproduce on a second GPU architecture.
6. Compare with equivalent public kernels such as GemLite/CUTLASS/QUICK where
   semantics can be matched.

This branch can be publishable even if its end-to-end gain on the already
optimized Qwen30 runtime is small, because the contribution is the verified
transformation and compiler, not one Amdahl-limited speedup.

## 3. TierFlow-R1: the fundamental model-design branch

Post-hoc inference research repeatedly found the same limit: frozen routers
create unstable external-memory traffic. TierFlow moves that constraint into
training.

### Minimal model

Start at 100M–300M parameters, not billions:

- 12–18 layers;
- 16 or 32 experts;
- top-2 or top-4;
- persistent route state;
- maximum route edits per token;
- progressive full-rank precision pages;
- cache/DMA state supplied to the controller;
- tier dropout.

### Primary equations

Stateful routing:

```text
R_l,t = Update(R_l,t-1, delta_R_l,t)
|delta_R_l,t| <= r
```

Progressive expert:

```text
W = W_2 + Delta_3 + Delta_4 + Delta_5
```

Hardware-aware objective:

```text
L = L_LM
  + lambda * E[T_critical]
  + mu * CVaR_0.95(T_critical)
  + eta * route_edits
  + rho * tier_dropout
```

Required baselines:

- standard MoE;
- StickyMoE;
- ReMoE;
- TriRoute or the closest reproducible joint controller;
- TierFlow route-only;
- TierFlow route + precision pages.

Hard gates:

- <=1% quality regression;
- >=4x fewer critical expert bytes;
- >=8x lower worst-case new expert loads;
- >=2x lower measured p95;
- no p99 burst collapse;
- transfer to a second simulated memory hierarchy.

This is the only branch that could change the future scaling law from
`bytes/token ~ top-k` toward `bytes/token ~ route edits + requested pages`.

## 4. What not to do

- Do not reopen CRAFT.
- Do not multiply unrelated local oracle gains.
- Do not spend more time on sub-2% P13 micro-integrations.
- Do not build Kimi K3 next.
- Do not assume MTP exists until a compatible artifact and runtime path are
  confirmed.
- Do not call the synthetic 80B gate a real 80B speed result.
- Do not buy 96 GB RAM before the full-size physical bank test.

## Decision

Run two projects in parallel:

```text
PORT80B  -> immediate 80B/8GB industry result
ERGV     -> formal exact-kernel science
```

Open TierFlow only as a separate training program with its own compute budget.

The next true Eureka is one of:

1. real Qwen3-Coder-Next 80B at >=10 tok/s on the same 8-GB laptop;
2. a general exact-reduction compiler that beats hand kernels across models;
3. a trained MoE whose external-memory traffic is structurally bounded.
