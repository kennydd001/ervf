# SweepSpec-50: first-principles path to 50 tokens/s

## Hard conclusion

A dense 27B model cannot reach 50 tokens/s on the locked laptop through
ordinary one-token autoregressive execution.

At roughly 4–4.5 bits, one complete 27B weight scan is approximately
15.6–17.1 GB. Even if every weight were resident in the GPU, a nominal
384-GB/s device-memory ceiling implies roughly 41–45 ms per weight sweep
before arithmetic and runtime overhead. With only 8 GB VRAM, the host-transfer
floor is much worse.

Therefore the only credible route to 50 tokens/s is:

```text
multiple accepted output tokens per target weight sweep
```

The primary systems metric becomes:

```text
AWS = accepted output tokens / target weight sweep
```

not merely draft acceptance rate or target calls.

## Two distinct targets

### LightningFlash-50 — immediate target

Use the existing Nemotron runtime, which already reaches approximately:

```text
short context: 21.4 tok/s
32K:           20.2 tok/s
128K:          16.7 tok/s
256K:          13.2 tok/s
```

To reach 50 tok/s, the required accepted tokens per expensive target round are
only about 2.34, 2.47, 3.00 and 3.78 respectively when drafting is hidden.

This is the highest-probability route.

### SweepSpec-Q38 — dense capability target

Qwen3.8-27B is dense. It needs roughly 20–25 accepted tokens per offloaded
target sweep on the locked hardware. Existing single-chain diffusion drafts
are insufficient by themselves. A large causal proposal forest and an
offload-aware verifier are required.

## Relevant current techniques

- DFlash / DFlare: one-pass block-diffusion drafting.
- Domino / DominoTree and JetSpec: parallel drafting with causal branch
  conditioning; JetSpec reaches accepted lengths around 10.7 in favorable
  reasoning workloads.
- DDTree, CaDDTree, LibraSpec and BlockPilot: tree/budget/length selection based
  on measured marginal benefit.
- D²SD: adds recovery branches near the likely first rejection boundary.
- Bole: exact tree verification for hybrid recurrent/full-attention models,
  avoiding branch-by-branch state materialization.
- SpecExec: demonstrates that offloaded targets can verify hundreds or
  thousands of candidates for close to one weight-streaming cost.
- SubSpec: constructs a low-bit substitute draft directly from offloaded target
  layers and reports very high acceptance lengths on consumer deployments.
- CATS: cascaded self-speculation under the same peak-device-memory budget.
- Saguaro / speculative-speculative decoding: prepares likely next-round
  speculations while the current target verification is still running.
- GOOSE and TAPS: deep reliable context spines and task-specialized drafters.
- MicroSpec: reduces the draft LM-head vocabulary cost.
- MemSpec: manages multiple drafters as a memory-resident working set on edge
  devices.

## New hypothesis 1 — Offload-Amortized Verification Law

For a target tree with n proposal nodes:

```text
T_verify(n) = T_weight_sweep + n * T_marginal
```

On an offloaded dense model, `T_weight_sweep` is dominant. Therefore the
optimal tree is much larger than in GPU-resident serving.

The throughput objective is:

```text
throughput(n) = E[accepted_tokens(n)] /
                (T_draft(n) + T_weight_sweep + n*T_marginal)
```

Extend the tree while:

```text
marginal accepted-token gain >
current throughput * marginal verification cost
```

This is an offload-specific extension of cost-aware tree and diffusion-length
selection. The first make-or-break experiment is to measure the complete target
verification curve at 1–1024 proposal nodes.

## New hypothesis 2 — Causal Diffusion Forest

Construct one anisotropic proposal forest from four sources:

1. a deep prompt/ngram spine;
2. native MTP continuations;
3. a causal-parallel JetSpec/Domino-style draft head;
4. D²SD recovery branches around predicted rejection boundaries.

The reliable sources receive depth. Weak sources receive breadth. A
task-confidence router selects specialized code, reasoning or chat drafters.

Goal for Nemotron:

```text
mean accepted depth >= 4
```

Goal for Qwen3.8:

```text
mean accepted depth >= 18
preferred >= 22
```

## New hypothesis 3 — Bole Weight-Sweep Verification

Use Bole's tree-closed-form recurrent-state representation as the target
verification primitive for Gated DeltaNet/Mamba-style layers.

Do not materialize one recurrent state per proposal node. Store token-level
state factors and reconstruct only the accepted state.

Combine this with layerwise target-weight streaming:

```text
load target layer once
verify the complete proposal forest for that layer
discard layer weights
continue
```

This is the essential bridge between hybrid state models and large offloaded
proposal trees.

## New hypothesis 4 — Substitute Shadow Cascade

Keep a low-bit substitute target in VRAM only as a draft/ranking model:

```text
resident Q2/Q3 substitute
        ↓
generate and rank a large forest
        ↓
offloaded Q4/Q5 target verifies the final forest exactly
```

The substitute is not judged by standalone CE. The relevant oracle is:

```text
recall of the true target-accepted branch
```

This follows the SubSpec/CATS principle but should be tested on Qwen3.8's
hybrid architecture and with large parallel trees.

A poor Q2 standalone model may still be an excellent branch-ranking proxy.

## New hypothesis 5 — Speculative-Speculative Overlap

Run the drafter concurrently with the target sweep:

```text
GPU/CPU target verifies round t
NPU/CPU/lightweight GPU stream drafts likely round t+1 outcomes
```

Precompute draft forests for the most probable acceptance/rejection outcomes.
If the realized outcome is in the prepared set, the next round begins without
draft latency.

Use Saguaro-style geometric fan-out and measure hit probability versus draft
compute.

## New hypothesis 6 — Certified Lazy LM Head

Large proposal trees make the 248K-token LM head expensive.

For greedy verification, build a clustered exact maximum-inner-product index.
For each vocabulary cluster, retain a provable upper bound:

```text
max_i in cluster w_i^T h
<= centroid^T h + residual_norm_bound * ||h||
```

Evaluate likely candidate tokens first. Prune a cluster only when its upper
bound is below the current best exact logit. Fall back to the full LM head for
ambiguous nodes.

This is different from approximate vocabulary pruning: the greedy result must
remain exact. The first oracle measures how many vocabulary rows require exact
evaluation per proposal node.

## New hypothesis 7 — CPU/GPU Cooperative Tree Verification

Do not copy every offloaded weight to the GPU. Partition large dense matrices
so CPU and GPU compute disjoint row blocks concurrently for the entire proposal
tree. Merge only activations.

Tree verification changes GEMV into a small GEMM, where the CPU can use DDR5
bandwidth and vector units much more effectively.

Compare:

1. all weight streaming to GPU;
2. CPU-only offloaded partitions;
3. simultaneous CPU/GPU row partition;
4. low-bit resident base plus CPU/GPU residual partition.

The selection is purely physical and must preserve target semantics.

## New hypothesis 8 — MTP-initialized DFlash/JetSpec

Qwen3.8 contains native multi-token-prediction parameters. Use them as:

- initialization for the parallel draft head;
- target-feature selectors;
- supervision for a block-diffusion adapter;
- a causal spine combined with diffusion recovery branches.

Because Qwen3.8 shares its architecture with Qwen3.6, initialize from published
Qwen3.6 DFlash/JetSpec-style draft heads where tensor shapes match, then
distill against Qwen3.8.

## Immediate experimental order

### Phase A — Nemotron50

1. Measure native MTP and prompt-lookup baselines.
2. Implement Bole-compatible tree verification.
3. Train/port one DFlash or JetSpec draft.
4. Add LibraSpec/CaDDTree budget control.
5. Add D²SD recovery branches.
6. Add Saguaro overlap.
7. Gate at 4K, 128K and 256K.

Pass:

```text
4K >= 50 tok/s
128K >= 40 tok/s
256K >= 30 tok/s
```

Stretch:

```text
256K >= 50 tok/s
```

### Phase B — Qwen3.8 verifier roofline

Before draft training, measure target verification at:

```text
1, 4, 8, 16, 32, 64, 128, 256, 512, 1024 nodes
```

Report:

- one weight-sweep time;
- marginal time/node;
- Gated DeltaNet tree state;
- tree-attention time;
- LM-head time;
- VRAM/RAM;
- exactness.

Hard go/no-go:

```text
256-node verify <= 1.5x one-token target sweep
```

### Phase C — Qwen3.8 draft hierarchy

1. native MTP;
2. GOOSE prompt/ngram tree;
3. training-free SubSpec/CATS shadow;
4. DFlash;
5. JetSpec/DominoTree;
6. combined causal diffusion forest.

Hard gate:

```text
mean accepted tokens >= 18
round latency / accepted token <= 20 ms
```

If acceptance remains below 12, 50 tok/s is not credible on the current
offload path.

## Claim boundary

The individual ingredients are prior art. The potentially new systems
intersection is:

> Offload-aware, hybrid-state, large-tree speculative decoding that maximizes
> accepted tokens per target weight sweep on a consumer GPU.

A defensible result would be:

```text
SweepSpec: dense 27B, 8 GB VRAM, exact Q4/Q5 target semantics,
50+ tok/s at batch 1
```

A separate, more likely first result is:

```text
LightningFlash: Nemotron 30B, 8 GB VRAM, 128K context,
50+ tok/s
```
