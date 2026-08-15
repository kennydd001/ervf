# Roofline Pivot N1–N5 — Exact Efficiency Before More Approximation

## 1. Executive verdict

The imported N1–N5 measurements change the TreeSweep-200 program in one decisive way:

> The current runtime is not only doing too much work. A large fraction of the work that is mathematically unavoidable is being executed far below the measured memory roofline.

This creates a new exact track that must run before closing the 50/100 tok/s targets and before treating an unoptimized tree verifier as the final 200 tok/s ceiling.

The exact-efficiency stack is:

\[
\boxed{
\text{graph-resident token program}
+\text{gather-free sparse downflow}
+\text{KV-head-group attention sweep}
+\text{roofline-oriented GEMV}
}
\]

No model weights, routes, activations or output semantics are intentionally changed on this track.

## 2. Imported evidence and claim status

All values below enter the pack as `USER_MEASURED_UNVERIFIED` until E0 reproduces them from immutable raw artifacts.

| ID | Imported result | Immediate interpretation |
|---|---|---|
| N5 | Streaming read roofline: 338.4 GB/s | Physical byte floor can now be measured rather than inferred from a datasheet. |
| N5 | Context 0 compulsory bytes: about 1,953 MiB; floor 6.05 ms; 165.2 tok/s ceiling | A one-token exact sweep cannot reach 200 tok/s even at the measured byte roofline. |
| N5 | Context 262.1K compulsory bytes: about 2,721 MiB; floor 8.43 ms; 118.6 tok/s ceiling | Long-context 100 tok/s is not excluded, but it is near the exact one-token byte floor. |
| N1 | Eager 36.714 ms → CUDA graph 28.023 ms | 8.691 ms, or 23.7% of the token, is removable issue/orchestration overhead in the tested path. |
| N2 | `gather_down_sparse` contributes 8.192 ms with only 0.040 drift; scan+gather is 67.2% of the down path | The sparse down path is dominated by materializing the sparse vector rather than by the useful down projection. |
| N2 | Gather in-loop ≈4.3 GB/s versus ≈25.05 GB/s isolated | The integrated gather is about six times below its own isolated implementation. |
| N4 | Attention time fits bytes with slope 21.48 ms/GB, intercept −0.033 ms and R²=0.9964 | Attention is overwhelmingly byte-bound in the measured regime. |
| N4 | At 262K, halving KV bytes changes 17.047 → 8.615 ms | The byte model is experimentally causal, not merely correlational. |
| N4 | Attention achieves 47.2 GB/s versus 338.4 GB/s | The kernel is about 7.2× below the device streaming roofline. |
| N5 | Critical GEMV achieves 81.4 GB/s versus 338.4 GB/s | The critical GEMV family is about 4.2× below the same roofline. |
| N3 | 91% ReLU² outputs are zero, but a sound rank-64 prefilter certifies only 0.01%; rank 64 contains 21.8% energy and the matrix rank is 1856 | Exact low-rank pre-gating is closed. Post-activation exact sparsity remains useful and motivates gather-free execution. |

## 3. Corrected physical interpretation

### 3.1 Exact one-token ceilings

For compulsory bytes \(B\) and measured streaming bandwidth \(R\):

\[
T_{\min}=\frac{B}{R},\qquad S_{\max}=\frac{R}{B}.
\]

The imported floors imply:

| Context | Byte floor | Exact one-token ceiling |
|---|---:|---:|
| 0 | 6.05 ms | 165.2 tok/s |
| ≈262K | 8.43 ms | 118.6 tok/s |

Consequences:

- 1,000 tok/s single-stream is excluded by the measured byte floor.
- 200 tok/s single-stream is also excluded for a one-output-token weight sweep.
- 100 tok/s is not physically excluded, but requires roughly 60–84% of the measured streaming roofline on essentially every compulsory byte.
- 50 tok/s requires roughly 30–42% of the measured roofline and is therefore an exact-efficiency target, not necessarily a speculative-decoding target.
- 200 tok/s still requires temporal amortization: multiple committed output tokens per expensive target-weight sweep.

### 3.2 A high-value integrated coincidence

The imported exact graph gain is:

\[
36.714-28.023=8.691\text{ ms}.
\]

The imported gather cost is approximately:

\[
8.192\text{ ms}.
\]

If an integrated candidate preserves the graph gain and removes most of the gather path, the first-order projection is:

\[
28.023-8.192=19.831\text{ ms}
\]

or:

\[
50.43\text{ tok/s}.
\]

This is a projection only. The shared contract forbids claiming it until one integrated physical run measures it. It nevertheless makes N1+N2 the highest-value immediate experiment pair.

## 4. Hypothesis E1 — Graph-Resident Token Program

### Claim

A complete decode token can be represented as one reusable device-resident execution graph whose topology is static while routes, token IDs, expert IDs, KV positions and pointers remain dynamic data.

CUDA Graphs amortize host-side setup and launch work by defining and instantiating the workflow once. CUDA also provides conditional graph nodes and device graph launch for bounded device-side control flow, although graph topology remains fixed and the supported node types are restricted.

Primary references:

- NVIDIA CUDA Graph Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- NVIDIA CUDA Graph best practices: <https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html>
- Mirage Persistent Kernel: <https://arxiv.org/abs/2512.22219>

### Variants

1. **Host-launched full-token graph** with fixed buffers and device-resident indirection arrays.
2. **Context-bucket graphs** for attention lengths, avoiding graph rebuild on every token.
3. **Device conditional graph** for dynamic expert/cache branches already represented as fixed subgraphs.
4. **Device tail-launch token loop** where sampling updates the token/KV position and launches the next token graph without a host round-trip.
5. **Persistent token fabric** only after graph capture: a small number of persistent operation-family kernels driven by device queues.

### Exactness controls

- same token IDs and logits as eager reference;
- same Mamba states, KV digests, routes and router weights;
- no graph-specific buffer aliasing;
- graph and eager paths read the same weight bytes;
- graph output compared over at least 10,000 token steps before integration claims.

### Gates

- reproduce 36.714 → approximately 28.023 ms within 5%;
- strong gate: ≤26 ms before gather/attention/GEMV changes;
- no larger VRAM high-watermark beyond the preregistered allowance;
- p95 improvement, not only mean improvement.

## 5. Hypothesis E2 — Gather-Free Sparse Downflow

### Observation

The exact ReLU² mask is available only after `fc1`, so N3 correctly closes low-rank pre-gating. But once the mask exists, writing a compact sparse activation buffer and reading it back is not mathematically necessary.

### Current conceptual path

```text
fc1 → ReLU² → panel scan → global gather buffer → sparse down GEMV
```

### Proposed path

```text
fc1 → ReLU² + bitmask → index-carrying down GEMV
```

The down kernel traverses the exact bitmask in deterministic index order, reads each surviving activation directly from the original ReLU² output and reads the corresponding logical weight column. No global compact activation tensor is materialized.

### Variants

1. **Ballot iterator:** warp ballot + popcount generates survivor indices in ascending logical order inside the down kernel.
2. **Panel-major down layout:** physically tile the down matrix by logical input-channel panels while retaining the original quantization-group IDs and reduction positions.
3. **Producer-consumer fusion:** `fc1/ReLU²` producer writes panel masks and values into a shared/ring buffer consumed directly by the down projection.
4. **Persistent expert CTA:** the same CTA or cooperative kernel performs ReLU² masking and down accumulation for one expert row block.
5. **No-compaction exact baseline:** process zeros directly but remove gather; this separates gather elimination from arithmetic sparsity.

### Numerical requirement

The fused kernel must reproduce the registered sparse-reference semantics. If bitwise identity is impossible because the reference changes batch/reduction order, the exact reduction graph and dtype boundaries must be frozen before testing, with a zero-additional-quality-regression gate.

### Performance gates

- eliminate at least 80% of the measured 8.192 ms gather cost;
- sparse down path at least 1.8× faster end to end;
- strong gate: full-token graph + gather-free path ≤20 ms at context 0;
- no hidden duplicated down-weight layout unless total bytes are included.

A fused gather–compute–scatter kernel has produced multi-fold speedups in other irregular GPU workloads by removing intermediate DRAM traffic, but that work is not an LLM/MoE result and is used only as a systems analogue: <https://arxiv.org/abs/2604.18020>.

## 6. Hypothesis E3 — KV-Head-Group Attention Sweep

### Observation

The attention fit is almost perfectly linear in KV bytes, but the measured kernel reaches only 47.2 GB/s. Therefore the first question is not whether attention can read fewer bytes; it is why the existing kernel converts compulsory bytes into only 14% of the measured streaming roofline.

### Proposed dataflow

For grouped-query attention, multiple query heads share one KV head. A kernel should:

1. schedule work by KV-head group;
2. load one K/V tile once;
3. broadcast that tile to all query heads in the group;
4. compute online softmax and value accumulation without score materialization;
5. use sequence splitting/work queues to expose enough parallel work at batch 1;
6. keep page-table and position logic out of the innermost load loop;
7. use double-buffered producer/consumer warps where supported.

Primary references:

- FlashInfer: <https://arxiv.org/abs/2501.01005>
- PersistentKV: <https://arxiv.org/abs/2606.26666>
- FlashAttention-3: <https://arxiv.org/abs/2407.08608>

FlashInfer explicitly specializes GQA decode and paged KV layouts. PersistentKV maps work by KV-head group and uses page-aware work queues. FlashAttention-3 demonstrates the general value of asynchronous producer/consumer pipelines and interleaving memory movement with softmax/matmul, though its reported hardware differs from the target laptop.

### Measurements before optimization

For context lengths 0/4K/32K/128K/262K:

- raw contiguous KV scan;
- actual address-pattern scan without math;
- score-only;
- softmax-only;
- value accumulation;
- full attention;
- global load efficiency, sectors/request, L1/L2 hit rate, occupancy and register spills.

### Gates

- reproduce the 21.48 ms/GB fit and R²≥0.99;
- first gate: ≥100 GB/s effective at 262K;
- strong gate: ≥169 GB/s, about 50% of the measured roofline;
- target attention ≤6 ms at 262K; stretch ≈4.8 ms;
- exact/tolerance-controlled output and identical KV commit.

## 7. Hypothesis E4 — Projection-Sweep GEMV

### Observation

Critical GEMV currently achieves about 81.4 GB/s, only 24% of the measured streaming roofline. Previous vector-load and reduction-autotuning wins were real component gains but diluted after integration; this branch must optimize the complete projection sweep, not an isolated inner loop.

### Proposed mechanisms

1. **Activation broadcast:** load each activation tile once into shared memory and let multiple output-row subwarps consume it.
2. **Scale broadcast:** decode one group scale once per relevant row bundle.
3. **Persistent row work queue:** balance heterogeneous row counts without many short launches.
4. **Exact virtual reduction multiplexing:** retain the successful reduction-graph autotuner while selecting lane width from an analytic register/occupancy model.
5. **Projection families:** jointly schedule same-input projections such as Q/K/V, gate/up, or architecture-specific Mamba projections without forcing a single monolithic quantization format.
6. **Producer/consumer stages:** overlap weight loads/decode with accumulation.
7. **Graph-resident pointers:** no CPU dispatch per matrix.

Mirage Persistent Kernel reports up to 1.7× end-to-end latency improvement from SM-level task graphs and persistent scheduling; it is strong prior art for the broad megakernel concept, not evidence that this target will obtain the same result: <https://arxiv.org/abs/2512.22219>.

### Gates

- reproduce 81.4 GB/s;
- first gate: ≥140 GB/s on the weighted critical-shape suite;
- strong gate: ≥170 GB/s;
- no regression larger than 5% on any registered critical matrix family;
- full-token improvement ≥8%, since smaller component-only gains have already failed integration gates.

## 8. Hypothesis E5 — Exact Persistent Token Fabric

This is the integrated hypothesis, not the sum of projected component percentages.

### Design

- one full-token CUDA graph or device-resident token loop;
- graph nodes consume dynamic route/index descriptors;
- gather-free sparse down;
- roofline-oriented attention and GEMV;
- no per-token allocations;
- fixed memory addresses and reusable staging;
- device-resident sampling/token update;
- exact state and cache mutation.

### Milestones

| Milestone | Gate |
|---|---:|
| E50 | ≤20.0 ms/token = ≥50 tok/s exact |
| E75 | ≤13.33 ms/token = ≥75 tok/s exact |
| E100 | ≤10.0 ms/token = ≥100 tok/s exact |

The ctx0 byte floor is 6.05 ms, so E100 leaves only 3.95 ms for all nonstream overhead. It is a stretch gate, not the expected first result.

### Required run

- batch 1;
- at least 10,000 causal output tokens for the final E50 claim;
- context 0/4K/32K and separate 262K profile;
- p50/p95/p99/max;
- exact target semantics;
- one-hour thermal run for any ≥50 tok/s claim;
- integrated physical timing only.

## 9. Consequence for TreeSweep-200

The old program hard-stopped if the baseline target verifier plus oracle proposal coverage stayed below 250 committed-equivalent tok/s. That is now too aggressive because N1–N5 show the baseline verifier may be dominated by recoverable implementation inefficiency.

The updated order is:

1. run coverage oracle — this is algorithmic and can hard-stop independently;
2. measure the baseline tree verifier;
3. reproduce and optimize E1–E5;
4. rerun the exact target-tree roofline with the frozen best exact-efficiency runtime;
5. only the **optimized** verifier plus coverage oracle may hard-stop the 200 tok/s track.

One-token 200 tok/s remains impossible under the measured 6.05 ms byte floor. TreeSweep is still necessary for 200, but its verifier must use the efficient target kernels rather than the current under-roofline path.

## 10. N3 closure

The exact low-rank ReLU² prefilter is closed unless a future experiment supplies a mechanistically independent certificate. The evidence is not merely a failed threshold: the relevant expert matrix is nearly full rank, rank 64 captures only 21.8% of energy and the sound certificate finds 0.01% of zeros despite 91% post-activation sparsity.

Do not reopen this as:

- a larger rank sweep;
- a learned unsafe predictor inside the exact track;
- an RSIV/GhostWeights variant;
- a local oracle without full-depth execution.

The reusable positive fact is only that exact post-ReLU² sparsity exists and should be consumed without a slow gather.

## 11. Updated breakthrough ladder

| Level | Meaning |
|---|---|
| 50 tok/s | Exact single-token efficiency breakthrough; no speculation required. |
| 100 tok/s | Near-roofline exact single-token execution at short context. |
| 200 tok/s | Requires TreeSweep/temporal amortization on top of the exact-efficiency runtime. |

A final 200 tok/s claim still requires one integrated candidate, ≥512 causal outputs, p95/p99, memory and thermal gates, and independent verification.
