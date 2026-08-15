# TreeSweep-200 — Causal Branch Compilation for Hybrid Mamba–Transformer MoE Models

## Executive verdict

A single-stream target of 200 tok/s is not reachable by merely extending the current linear MTP chain or by shaving another 10–30% from target weight traffic. It requires an architectural change in how one target-model call is used.

The proposed program is:

\[
\boxed{
\text{high-coverage candidate tree}
+\text{one exact hybrid target sweep}
+\text{shared Mamba/GQA/MoE work}
}
\]

The system name is **TreeSweep-200**. The scientific formulation is **Causal Branch Compilation (CBC)**:

> Compile a candidate prefix tree into a hardware-aware execution graph that shares target weights, recurrent-state transitions, prefix KV, and expert records across branches while preserving exact target semantics.

This is a hypothesis. The pack places two cheap oracles before implementation:

1. Can the target verify a tree quickly enough?
2. Can a small tree cover the target path deeply enough?

If either answer is no, 200 tok/s is closed without building a new drafter.

---


## 0.5 N1–N5 exact-efficiency pivot

A later measurement round found that the current decoder executes compulsory bytes far below an independently measured 338.4 GB/s streaming roofline. The imported results include a 23.7% CUDA-graph gain, an 8.192 ms sparse-down gather, attention at 47.2 GB/s and critical GEMV at 81.4 GB/s. These values are not accepted until Agent 18 reproduces them.

The corrected physical picture is:

- exact one-token ctx0 floor ≈6.05 ms, ceiling ≈165.2 tok/s;
- exact one-token ~262K floor ≈8.43 ms, ceiling ≈118.6 tok/s;
- 50 and possibly 100 tok/s may be exact-efficiency problems;
- 200 tok/s remains impossible for one output per target sweep and therefore still requires TreeSweep/temporal amortization.

The pack now has a parallel exact-efficiency track described in `ROOFLINE_PIVOT_N1_N5_REPORT_2026-08-15.md`. A baseline tree verifier may no longer hard-close the 200 tok/s track. Only the optimized rerun after graph/gather/attention/GEMV work can do so.

## 1. Current provisional baseline

The existing EXACTFLOW report records provisional values that P0 must remeasure:

- short autoregressive decode: 27.743 tok/s;
- short token latency: 36.045 ms;
- mean accepted draft tokens: 2.114;
- mean output tokens per target round: 3.114.

Thus:

\[
\text{throughput}
=\frac{A}{T_d+T_v}
\]

where `A` is output tokens per round, `T_d` draft time and `T_v` target verification time.

For 200 tok/s:

\[
T_{round}\leq 5A\;\text{ms}.
\]

| Output tokens/round | Maximum draft+verify round |
|---:|---:|
| 3.114 | 15.57 ms |
| 5 | 25.00 ms |
| 8 | 40.00 ms |
| 10 | 50.00 ms |
| 12 | 60.00 ms |

The plausible operating points are therefore not “3.114 tokens in 15.57 ms.” They are closer to:

- 8 output tokens in 40 ms;
- 10 output tokens in 50 ms;
- 12 output tokens in 60 ms.

---

## 2. Why a longer linear chain is insufficient

As a diagnostic approximation, assume every extra draft position has the same conditional acceptance probability `p`. The measured accepted drafts satisfy:

\[
p+p^2+p^3+p^4+p^5=2.114.
\]

Solving gives:

\[
p\approx0.725823.
\]

The infinite-chain expectation is:

\[
\sum_{i=1}^{\infty}p^i
=\frac{p}{1-p}
\approx2.647.
\]

Including the guaranteed target token gives only:

\[
A_{\infty}\approx3.647.
\]

This is not a theorem about the real non-stationary MTP head. It is a strong diagnostic: keeping the same error process and drafting 16 or 32 tokens will not approach the 8–10 output tokens required for 200 tok/s.

Therefore at least one must change:

1. conditional draft quality rises toward roughly 90–95%;
2. multiple alternatives are retained in a tree so an early wrong top-1 draft does not destroy deeper correct continuations;
3. the drafter predicts dependent future tokens in one parallel call rather than an autoregressive chain;
4. a diffusion/block drafter produces long, coherent proposals in parallel.

---

## 3. The two-dimensional speed-of-light test

Tree speculation succeeds only if **coverage** and **verification cost** align.

Let:

- `N` = number of candidate tree nodes;
- `D` = accepted depth on the realized target path;
- `A = D + 1` = output tokens per round in greedy decoding;
- `T_v(N, topology, context)` = exact target tree-verification time;
- `T_d` = draft time.

Then:

\[
\text{tok/s}=\frac{D+1}{T_d+T_v}.
\]

### Oracle A — target verifier roofline

Use known candidate tokens and verify trees of 1, 5, 15, 31 and 63 nodes. Measure the exact physical runtime of:

- Mamba tree scan;
- GQA tree attention;
- natural MoE routing and expert union;
- LM-head and acceptance;
- state commit/replay.

### Oracle B — target-informed coverage

Use target distributions to construct the best possible tree under node budgets 8, 16, 32 and 64. This is an unattainable upper bound for any real drafter, but it answers whether the vocabulary distribution itself permits deep coverage.

### Joint oracle ceiling

For each budget:

\[
S_{oracle}(N)=\frac{A_{oracle}(N)}{T_v(N)}.
\]

The strong gate is:

\[
\max_{N\le64}S_{oracle}(N)\ge250\text{ tok/s}.
\]

The 250 tok/s margin leaves approximately 20% room for drafting and orchestration. If this gate fails, single-stream 200 tok/s is physically blocked for this target/hardware pair.

---

## 4. TreeSweep-200 architecture

### 4.1 Candidate generation

The proposal side may use one of several branches:

- native MTP logits arranged as a dynamic tree;
- Hydra/FastMTP-style dependent heads;
- EAGLE-3-style direct-token drafter with multi-layer features;
- Parallel Token Prediction (PTP), which jointly models dependent future tokens;
- a small diffusion drafter;
- Nemotron Elastic 12B/23B as a conditional drafter;
- training-free n-gram/lookahead paths for code and repetitive text.

The target remains authoritative.

### 4.2 Causal Branch Compiler

Given tree `T`, compile an execution schedule minimizing physical critical-path time:

\[
T(T)=T_d(T)+T_{Mamba}(T)+T_{GQA}(T)+T_{MoE}(T)+T_{head}(T)+T_{sched}(T).
\]

The tree objective is:

\[
\max_T
\frac{\mathbb E[A(T)]}{T(T)}.
\]

This differs from tree optimizers that maximize acceptance under a pure node budget. TreeSweep includes:

- Mamba-state transition cost;
- GQA/KV tree-mask cost;
- exact target expert-union/H2D cost;
- target weight reuse;
- draft cost;
- VRAM/state footprint;
- p95 critical-path penalties.

### 4.3 Exact Mamba tree scan

Nemotron interleaves Mamba-2 layers. Naively unrolling every tree path duplicates prefix computations and requires many recurrent states.

STree shows that SSM state transitions can be accumulated according to a prefix-tree topology. TreeSweep must adapt that principle to the exact Nemotron Mamba-2 implementation:

- pack tree nodes once;
- compute accumulated transition products/sums along parent paths;
- retain branch-local temporary states only as needed;
- commit only the accepted path;
- use activation replay or exact accepted-state reconstruction after rejection.

All results must match sequential target evaluation within the registered numerical reference.

### 4.4 Packed GQA tree attention

The six GQA layers use a topology-aware mask:

- every node sees the permanent prefix;
- every node sees only its ancestors;
- siblings do not see one another;
- K/V for shared prefixes are loaded once;
- branch K/V is temporary;
- only the accepted branch is committed.

LongSpec-style hybrid attention can combine an optimized prefix path with a smaller tree-mask path.

### 4.5 Exact expert-major MoE verification

Every tree node uses the target router on its own target hidden state. No expert dropping is allowed on the exact track.

For each MoE layer:

1. route all tree nodes;
2. group `(node, expert)` pairs by expert ID;
3. load each active expert record once per round;
4. process all assigned node rows in a grouped GEMM/GEMV;
5. apply the original router weights;
6. scatter results back to tree nodes.

The relevant physical cost is the **expert union**, not merely node count.

Let `U_l(T)` be the set of natural target experts activated by tree `T` in layer `l`. An approximate cost term is:

\[
T_{MoE}(T)\approx
\sum_l
\max\left(
T^{compute}_l(T),
\frac{|U_l(T)\setminus Cache_l|S_e}{B_{H2D}}
\right).
\]

The tree constructor may predict this cost, but target routing and final verification remain exact.

### 4.6 Exact commit

After target logits are produced:

- determine accepted prefix with the frozen verification rule;
- commit target tokens only;
- commit GQA KV only for that prefix;
- reconstruct/commit the corresponding Mamba states;
- discard all siblings and rejected suffixes;
- preserve cache accounting separately from semantic state.

---

## 5. Hypothesis family

### H1 — Hybrid target trees are cheap enough

A 15–63-node tree can be verified at much less than `N` times one-token latency because target weights are reused and prefix work is shared.

**Falsification:** the joint verifier/coverage ceiling stays below 250 tok/s.

### H2 — Dynamic tree coverage beats the MTP chain ceiling

A dynamic tree using top-k alternatives at uncertain positions can achieve accepted depth ≥7 with ≤32 nodes or depth ≥9 with ≤64 nodes.

**Falsification:** target-informed oracle misses both gates.

### H3 — Mamba tree scan prevents recurrent-state explosion

Accumulated Mamba-2 transitions can evaluate packed branch states with modest overhead relative to one chain.

**Falsification:** tree scan is slower than unrolled verification or cannot reproduce accepted states.

### H4 — Natural-route expert-major batching keeps MoE union manageable

Exact grouping can amortize each expert record across multiple tree nodes, and a cost-aware tree can avoid unnecessary expert scattering without changing target routing.

**Falsification:** natural target expert union/H2D dominates and no tree meeting coverage gates fits the round budget.

### H5 — Native MTP contains more useful probability mass than its top-1 chain reveals

Top-k alternatives from native MTP heads may form a useful tree even if the top-1 chain acceptance is only 2.114 drafts.

**Falsification:** native MTP tree underperforms the target-informed oracle by too much to meet real throughput gates.

### H6 — One-shot dependent drafting can break the chain ceiling

PTP, FastEagle, FastMTP or a Hydra-like lightweight adapter can jointly propose 8–12 dependent future tokens in one call.

**Falsification:** integrated output/ms does not beat native MTP after training and draft cost.

### H7 — Diffusion drafting offers the highest acceptance ceiling

A small aligned diffusion drafter may propose a long block or prefix tree in one/few passes. DEER reports acceptance lengths up to 32 on other targets, establishing feasibility but not transfer to Nemotron.

**Falsification:** draft cost plus target verification misses the round gate, or training/resources exceed the registered budget.

### H8 — BranchCert can reduce off-path precision exactly

Off-path tree nodes may be evaluated from a low-bit core and exact residual bounds. Fetch full NVFP4 pages only when intervals cannot certify the acceptance decision or rounded state.

This is an exact optional modifier, not a learned risk gate.

**Falsification:** certificate pass rate or saved time misses its oracle gate.

### H9 — OrbitANS lowers one target sweep exactly

Random-access entropy coding of NVFP4 codes/scales may reduce transferred bytes without changing target semantics.

**Falsification:** physical savings after decode overhead are below 5%.

### H10 — PathQ creates a faster quality-controlled target

Mixed precision per layer/expert/projection can minimize active bytes rather than file size. This is a separate, lossy registry.

**Falsification:** test quality or physical speed gates fail.

### H11 — Heterogeneous drafting can hide draft time

A CPU/iGPU/NPU drafter can run asynchronously while the dGPU target verifies the previous tree.

**Falsification:** measured device/runtime overhead cannot be hidden or increases critical-path time.

### H12 — Long-context tree verification remains viable

Constant-memory drafting, prefix-optimized attention and quantized draft KV can preserve a useful tree advantage at 128K/262K.

**Falsification:** tree/KV attention dominates and the context throughput gates fail.

---

## 6. Draft branches

### 6.1 Native MTP tree

Before training anything:

- capture top-k logits for every MTP depth;
- build fixed and dynamic trees;
- test consistency between depth heads;
- score with Sequoia, OPT-Tree and DySpec-style objectives;
- measure real draft cost and target coverage.

This is the cheapest branch.

### 6.2 Sequentially dependent heads

Hydra and FastMTP indicate that conditioning later draft positions on earlier proposed tokens materially improves acceptance. A small adapter may be trained over frozen target features.

Required comparisons:

- independent MTP heads;
- dependent recurrent/head cascade;
- direct token prediction using fused features;
- same parameter/training budget.

### 6.3 Parallel Token Prediction

PTP explicitly models dependencies among several future tokens in one call. It can be trained by distillation from the target. The relevant gate is not paper speedup but whether the drafter yields a high-coverage ≤32/64-node tree at low enough cost.

### 6.4 Diffusion drafter

Potential options:

- a small denoiser trained against Lightning target trajectories;
- Nemotron-TwoTower denoiser as a research teacher;
- DEER/PRESTO-style diffusion prefix trees;
- Nemotron-Labs-Diffusion tri-mode concepts.

This branch is training-heavy and must follow the cheap oracles.

### 6.5 Elastic nested drafter

The public Nemotron Elastic checkpoint contains nested 30B/23B/12B variants. It is useful only if:

- artifact/tokenizer compatibility with the target is locked;
- draft latency is low enough;
- shared storage actually reduces the working set;
- integrated throughput improves.

---

## 7. Exact modifiers carried forward

### OrbitANS

Compress official NVFP4 codes and FP8 scales exactly with random-access tile coding and fused decode. Use only after full-bank census and physical savings gate.

### CertiPlane / BranchCert

Represent target weights as core plus exact pages. Use conservative intervals to determine when omitted pages cannot change:

- router top-k;
- ReLU² sign/rounded activation;
- rounded projection output;
- acceptance comparison.

Off-path nodes are a particularly promising target because most are discarded.

### PathQ

Optimize expected active bytes and critical-path latency under a frozen quality budget. Limit the number of formats on the critical path. Never describe it as exact.

---

## 8. Critical experiments and gates

### P0 — Identity and baseline

Must classify “Nemotron 3.5 Lightning” relative to public Nano/NIM/Elastic artifacts and reproduce measurements.

### P1 — Target-only tree verifier

Tree sizes: 1, 5, 15, 31, 63. Contexts: short, 128K, 262K where feasible.

Outputs:

- exactness controls;
- `T_v(N)` by component;
- expert union/H2D;
- state/KV memory;
- fit `C+aN+bN²`.

### P2 — Oracle tree coverage

Node budgets: 8, 16, 32, 64. Measure target-path depth and exact accepted output.

Primary gates:

- depth ≥7 with ≤32 nodes;
- or depth ≥9 with ≤64 nodes.

### P2C — Joint speed of light

\[
\max_N A_{oracle}(N)/T_v(N)\ge250\text{ tok/s}.
\]

Failure closes 200 tok/s exact track.

### P3–P5 — Hybrid target kernels

- MambaTree exact scan;
- topology-aware GQA tree attention;
- expert-major natural-route verifier.

### P6 — Native MTP tree

Must beat linear MTP on output/ms.

### P7/P8 — Learned parallel/diffusion drafters

Open only after P2C passes.

### P9 — Integrated 100 tok/s milestone

Useful intermediate gate; not the final claim.

### P10 — Integrated 200 tok/s

One frozen physical candidate:

- ≥200 tok/s short-context single-stream;
- 8 GiB VRAM;
- 64 GiB research RAM, 32 GiB product stretch;
- ≥512 causal outputs;
- exact target semantics or separately registered quality target;
- p95/p99 and one-hour thermal run;
- independent verification.

---

## 9. Novelty boundary

Tree speculative decoding, hardware-aware tree search, SSM tree scans, MoE expert-union optimization, diffusion drafting, MTP heads and asynchronous drafting all have direct prior art.

The potentially distinct intersection is:

> An exact Causal Branch Compiler for a hybrid Mamba-2/GQA/MoE target that jointly optimizes candidate-tree coverage, accumulated SSM transitions, topology-aware KV, natural target expert-union traffic, exact precision certification and physical round latency on a memory-constrained consumer GPU.

This is negative search evidence only. A formal novelty and patent audit remains mandatory after a technical candidate passes.

---

## 10. Expected decision outcomes

### Outcome A — verifier/coverage oracle fails

The 200 tok/s target is falsified on this hardware/target. Continue optimizing toward 50–100 tok/s, but do not train a new drafter for 200.

### Outcome B — oracle passes, native MTP tree fails

Train a dependent parallel drafter (PTP/FastEagle/EAGLE-style) or diffusion drafter.

### Outcome C — native MTP tree passes

Integrate immediately; do not add training complexity.

### Outcome D — short-context 200 passes, long context fails

Open LongSpec/QuantSpec-style KV and draft-memory work as a separate long-context program.

### Outcome E — exact track near-misses on round time

Use OrbitANS or BranchCert. Do not change target quality unless an independent PathQ registry is approved.

---

## 11. Final position

The project is now asking the right question:

> Can one expensive target weightsweep emit eight to ten authoritative tokens rather than one?

If yes, 200 tok/s becomes a matter of candidate coverage and exact hybrid tree compilation. If no, no amount of incremental quantization or single-token kernel tuning will close the gap.


## 13. Exact-efficiency companion hypothesis

TreeSweep is now paired with an exact single-token program:

\[
oxed{
	ext{GraphFlow}+	ext{GatherlessDown}+	ext{KVGroupSweep}+	ext{ProjectionSweep}
}
\]

Its milestones are 50/75/100 tok/s. A successful exact-efficiency runtime is then used as the target verifier inside TreeSweep. A failed E50 does not by itself falsify TreeSweep, but it lowers the plausible optimized verifier ceiling.
