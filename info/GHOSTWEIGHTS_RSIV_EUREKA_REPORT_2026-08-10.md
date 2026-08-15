# GhostWeights / RSIV-MoE
## Routed Subspace Image Virtualization for prompt-compiled, disk-backed LLM inference

**Date:** 2026-08-10  
**Status:** mathematically grounded Eureka hypothesis; not yet an experimentally demonstrated breakthrough  
**Starting point:** the completed CRAFT-MoE program is frozen as `closed_no_eureka`. This proposal is deliberately mechanistically independent.

---

## 1. Executive verdict

CRAFT-MoE tested shared surrogates, route substitution, route×precision programs, neuron/atom sparsity, residual sketches, block coalescing, co-routed error cancellation, cache-span reconstruction and reduction-order effects. The strongest local oracles did not stack through all layers. The recurring failure was cumulative state drift: a locally cheap approximation altered the hidden trajectory, after which later routing and transformations changed.

The new hypothesis does **not** prune experts, replace routes, approximate the whole expert with a student, or assume low-rank expert weights.

It treats the full model weights as a **cold backing store** and keeps only the model's action on the activation directions actually visited by the current workload. The resident object is therefore not the model checkpoint, but an **operator-image working set**.

> **Eureka thesis:** for inference, the resident information requirement of a routed MoE can scale with the rank of the routed activation workload, rather than with the number or total size of experts.

The name used in this report is:

> **RSIV-MoE — Routed Subspace Image Virtualization**  
> Product codename: **GhostWeights**

---

## 2. The linear operator identity

For a linear map

\[
y = W x, \qquad W\in\mathbb{R}^{d_{out}\times d},
\]

let \(Q\in\mathbb{R}^{d\times r}\) be an orthonormal basis of a workload-specific activation subspace. Decompose

\[
x = Q Q^\top x + r_x.
\]

Precompute the **operator image**

\[
M = WQ.
\]

Then

\[
Wx = M(Q^\top x) + Wr_x.
\]

If \(x\in\mathrm{span}(Q)\), then \(r_x=0\) and the reduced computation is exact:

\[
Wx = M(Q^\top x).
\]

The resident map contains \(d r + d_{out}r\) elements instead of \(d d_{out}\). A cold-path invocation is needed only when the current activation has a significant component outside the cached subspace.

### Online operator memoization

When a new residual direction appears,

\[
v = r_x/\|r_x\|,
\]

the slow path computes \(Wv\) once and appends

\[
Q\leftarrow[Q,v], \qquad M\leftarrow[M,Wv].
\]

Every previous direction remains exact. The number of cold operator expansions over a sequence is bounded by the rank growth of the query matrix, not by the number of matrix-vector calls.

---

## 3. Compiling a SwiGLU expert

For routed expert \(e\):

\[
g = G_e x,
\]

\[
u = U_e x,
\]

\[
z = \operatorname{SiLU}(g)\odot u,
\]

\[
y = D_e z.
\]

Maintain two routed subspaces:

- input basis \(Q_e\in\mathbb{R}^{d\times r_e}\);
- intermediate basis \(P_e\in\mathbb{R}^{m\times s_e}\).

Cache the operator images

\[
A_e=G_eQ_e,\qquad B_e=U_eQ_e,\qquad C_e=D_eP_e.
\]

The fast path is

\[
q=Q_e^\top x,
\]

\[
\hat g=A_eq,\qquad \hat u=B_eq,
\]

\[
\hat z=\operatorname{SiLU}(\hat g)\odot\hat u,
\]

\[
p=P_e^\top\hat z,
\]

\[
\hat y=C_ep.
\]

There are two independent page-fault gates:

\[
\rho_x=\frac{\|x-Q_eQ_e^\top x\|}{\|x\|},
\]

\[
\rho_z=\frac{\|\hat z-P_eP_e^\top\hat z\|}{\|\hat z\|}.
\]

This permits staged fallback:

1. If \(\rho_x\) is small, `gate` and `up` use the image cache.
2. If \(\rho_x\) is large, load only \(G_e,U_e\), compute exactly and expand \(Q_e,A_e,B_e\).
3. If \(\rho_z\) is small, `down` uses the image cache.
4. If \(\rho_z\) is large, load only \(D_e\), compute exactly and expand \(P_e,C_e\).

The method therefore avoids the all-or-nothing requirement to fetch a complete expert on every miss.

---

## 4. Prompt compilation theorem

For one expert, collect all routed prompt inputs in

\[
X_e=[x_1,\ldots,x_{n_e}]
\]

and all corresponding SwiGLU intermediates in

\[
Z_e=[z_1,\ldots,z_{n_e}].
\]

Let \(Q_e\) span \(X_e\), and let \(P_e\) span \(Z_e\). Then the compiled representation

\[
\{Q_e,G_eQ_e,U_eQ_e,P_e,D_eP_e\}
\]

reproduces the expert exactly on every prompt activation used to construct those spaces, up to the chosen numerical precision.

Crucially, these images can be built during normal prefill. If

\[
X_e=Q_eR_e
\]

and the already computed gate outputs are

\[
Y^g_e=G_eX_e,
\]

then

\[
G_eQ_e=Y^g_eR_e^\dagger.
\]

The same holds for `up` and `down`. Thus a prompt can compile its expert images from activation-response pairs while the original expert is being streamed for prefill. No second full-weight pass is fundamentally required.

---

## 5. The expert-count cancellation law

Consider one MoE layer with:

- \(E\) experts;
- top-\(k\) routing;
- prompt length \(T\);
- hidden width \(d\);
- expert intermediate width \(m\).

Expert \(e\) receives \(n_e\) prompt activations. Therefore

\[
r_e=\operatorname{rank}(X_e)\le n_e,
\]

\[
s_e=\operatorname{rank}(Z_e)\le n_e,
\]

and because exactly \(k\) experts are selected per token,

\[
\sum_{e=1}^{E}n_e=kT.
\]

The total number of elements in all exact prompt-compiled expert images is

\[
S=\sum_e[(d+2m)r_e+(m+d)s_e].
\]

Using \(r_e,s_e\le n_e\):

\[
\boxed{S\le(2d+3m)kT.}
\]

**The number of experts \(E\) cancels.**

This is the central mathematical result. At fixed prompt length and top-\(k\), the exact prompt working set is bounded by routed activation mass rather than total expert count.

### Consequence

Under balanced routing, average prompt observations per expert are

\[
\bar n=\frac{kT}{E}.
\]

For a 1,024-token prompt:

| Model | Experts/layer | Top-k | Average prompt inputs/expert |
|---|---:|---:|---:|
| DeepSeek-V2-Lite | 64 | 6 | 96.0 |
| DeepSeek-V4-Flash | 256 | 6 | 24.0 |
| Kimi K3 | 896 | 16 | 18.29 |

Counterintuitively, the very large expert bank of K3 gives each expert a smaller prompt-local query set than V2-Lite.

---

## 6. Exact prompt working-set bounds

The following values use the exact upper bound above and count one byte per cached element for an FP8-style representation. BF16 doubles them. Actual rank deficiency can reduce them further.

| Model | Prompt T | FP8 upper bound | BF16 upper bound |
|---|---:|---:|---:|
| V2-Lite | 1,024 | 1.238 GiB | 2.476 GiB |
| V4-Flash | 1,024 | 3.527 GiB | 7.055 GiB |
| Kimi K3 | 1,024 | 23.000 GiB | 46.000 GiB |
| V2-Lite | 4,096 | 4.951 GiB | 9.902 GiB |
| V4-Flash | 4,096 | 14.109 GiB | 28.219 GiB |
| Kimi K3 | 4,096 | 92.000 GiB | 184.000 GiB |

This is not yet the total model; dense attention/shared/trunk maps need the same operator-image treatment. It does show that a 1,024-token K3 expert working set is no longer automatically terabyte-scale.

---

## 7. Rank-capped universal atlas sizes

A deployable model can ship an offline calibration atlas for every expert, then add prompt-specific delta directions online.

For rank \(r=s\), image-cache elements per expert are

\[
(2d+3m)r.
\]

A mixed layout stores bases \(Q,P\) in BF16 and images \(A,B,C\) in INT8.

| Model / rank | All-FP8 atlas | Mixed BF16-basis/INT8-image atlas | Active mixed bytes/token |
|---|---:|---:|---:|
| V2-Lite, r=32 | 0.413 GiB | 0.584 GiB | 56.1 MiB |
| V4-Flash, r=16 | 2.352 GiB | 3.359 GiB | 80.6 MiB |
| V4-Flash, r=32 | 4.703 GiB | 6.719 GiB | 161.3 MiB |
| Kimi K3, r=8 | 10.063 GiB | 14.150 GiB | 258.8 MiB |
| Kimi K3, r=16 | 20.125 GiB | 28.301 GiB | 517.5 MiB |

At ten tokens per second, the K3 rank-8 fast path moves about 2.6 GiB/s of mixed image data before trunk, attention and other overhead. That is physically plausible on ordinary RAM/PCIe systems; the unknown is the cold-path rate.

---

## 8. Required fast-path fraction

Let \(c\) be the full-expert-byte to image-byte ratio and \(p\) the fast-path fraction. Ignoring overlap and compute, expert-read speedup is

\[
S=\frac{1}{p/c+(1-p)}.
\]

Approximate ratios:

- V4-Flash, native ~4.25-bit versus mixed rank-16 atlas: \(c\approx40.8\).
- K3, native ~4.25-bit versus mixed rank-8 atlas: \(c\approx95.2\).

For a 10× expert-read reduction:

\[
p\ge\frac{0.9}{1-1/c}.
\]

This gives:

- V4 rank-16: **p ≥ 92.3%**;
- K3 rank-8: **p ≥ 91.0%**.

For acceptable p95 latency on SSD-backed misses, the practical requirement will probably be stricter—likely 97–99% or effective asynchronous coverage.

---

## 9. A stronger three-tier residual design

The basic fast/slow split can be improved:

### Hot tier

High-quality operator images for the common activation subspaces.

### Warm tier

A very low-bit full operator \(Q_b(W)\), or selected cold residual pages.

For

\[
x=Qq+r,
\]

use

\[
\hat y=(WQ)q+Q_b(W)r.
\]

Then

\[
y-\hat y=(W-Q_b(W))r,
\]

and

\[
\|y-\hat y\|\le\|W-Q_b(W)\|_2\|r\|.
\]

The low-bit quantizer is applied only to the low-energy off-manifold residual, not to the dominant activation component. This is materially different from the model-wide Q2/Q3 runs that failed in CRAFT.

### Cold tier

Full or higher-bit weights on NVMe/remote storage for rare large residuals, atlas expansion and exact safety fallback.

---

## 10. Why this survives the completed CRAFT falsifications

| CRAFT failure | Why it does not directly falsify RSIV |
|---|---|
| Shared output bases had high oracle error | RSIV does not place outputs from many experts in one shared output basis. It stores each linear operator's exact action on its own routed input directions. |
| Shared/nonlinear surrogates lost to quantization | RSIV is not a learned replacement function. The image columns are derived directly from the original weights or exact activation-response pairs. |
| Route×bit CRCQ failed downstream | RSIV normally preserves the natural route and the original expert function on cached directions. |
| 10–25% atom sparsity failed full-depth | RSIV does not discard neuron contributions. It changes the representation used to compute them. |
| SketchGate missed damaging cases | RSIV's primary gate is a directly measured subspace residual. A slow path remains available. |
| Error cancellation and reduction order failed | RSIV does not rely on accidental cancellation or floating-point order effects. |
| Cache-span added little over zero-fill | RSIV caches an operator's response basis, not a linear combination of other experts' current outputs. |

The critical remaining risk is still full-depth trajectory stability. It must be tested causally on each policy's own hidden states.

---

## 11. Relation to existing work

### Closest mechanism: Gated Subspace Inference / Skyline Subspace Inference

GSI caches \(M=WV\) for dense transformer maps and uses a residual gate. It reports substantial linear-weight-read speedups on GPT-2, GPT-J and OPT. However, the paper stores the cached images **alongside the full weight matrices in HBM**. Its objective is dense-model acceleration, not capacity virtualization.

RSIV changes the systems contract:

1. full expert weights are a cold backing store, not resident state;
2. bases are routed per expert or per local chart;
3. both SwiGLU input and down-intermediate manifolds are represented;
4. prompt prefill compiles exact response images;
5. online page faults expand the atlas;
6. the primary metric is cold bytes/token and resident working-set size, not only HBM linear-layer speed.

### Existing MoE offloading

Known systems predict, cache, prefetch, substitute or load full/partial experts. The bounded search conducted for this report did not find a primary source implementing per-expert routed activation-subspace images as the resident replacement for disk-backed full experts. This is negative search evidence only, not a patentability or novelty conclusion.

### Important counterevidence

Recent MoE geometry work reports that expert-local Jacobians can have faster spectral decay while routed hidden representations can distribute variance over more directions. That makes the rank census and future-token fast-path rate the decisive experiment.

### Research-quality caveat

The GSI result is a recent arXiv preprint and no official public code was located in the bounded search. It should be independently reproduced rather than treated as settled fact.

---

## 12. Preregistered experiment sequence

### P0 — Freeze CRAFT and open an independent registry item

No CRAFT threshold, candidate or failed gate is changed. The new hypothesis is `RSIV_MOE_V1`.

### P1 — Routed rank census, no expert approximation yet

Use existing V2-Lite traces where possible.

For every layer/expert:

- build causal and offline \(Q_e\) bases from routed inputs;
- measure held-out residual ratios for ranks 4, 8, 16, 32, 64, 128;
- measure online rank growth and prompt→decode transfer;
- repeat for the exact SwiGLU intermediate \(z\) to obtain \(P_e\);
- report weighted p50/p95/p99 ranks and double-gate fast-path fraction.

Primary screen:

- rank cap ≤32;
- ≥92% of routed invocations pass both gates at a threshold that later preserves quality;
- projected routed cold-byte reduction ≥10× versus packed int4.

A V2 failure is informative but not automatically terminal because \(kT/E\) is materially smaller for V4/K3.

### P2 — Exact one-layer operator-image oracle

Construct real \(A_e,B_e,C_e\) from the original expert matrices.

Controls:

- exact prompt reconstruction at full numerical precision;
- direct comparison with ordinary SVD/output-basis compression;
- separate gate/up and down miss attribution;
- image quantization ablations BF16, FP8, INT8.

### P3 — Full-depth causal simulator

All 26 V2 MoE layers use their own states and routes.

On a gate miss:

- compute the original expert exactly;
- add the new residual direction and image column;
- record cold bytes and latency model.

Gates:

- relative CE increase <2%;
- no generation collapse;
- ≥10× routed cold-byte reduction;
- slow-path expert invocation rate ≤8%, with a secondary 3% tail-latency target;
- exact 100%-fallback control.

### P4 — Prompt compilation and autoregressive validation

At least 20 prompts across:

- code;
- arithmetic/reasoning;
- Dutch;
- ordinary English text;
- long-context retrieval.

Each prompt: ≥512 generated tokens.

Measure:

- basis rank growth after prefill;
- page faults per emitted token;
- p50/p95/p99 cold bytes;
- token agreement, KL and task accuracy;
- atlas reuse across conversation turns.

### P5 — Packed runtime

Implement:

- BF16/FP8/INT8 basis/image layouts;
- atomically updatable QR/DGKS bases;
- partial gate/up/down fallback;
- pinned-host staging and async NVMe prefetch;
- optional Q2 residual tier;
- shared byte-budget across layers.

Gate:

- ≥2× actual batch-1 decode speed versus a real packed-int4 baseline on the target laptop;
- no unacceptable p95 stalls.

### P6 — Second architecture

Use a higher-expert-count model, preferably Qwen3-30B-A3B or Kimi-Linear-48B-A3B. This is required because RSIV predicts more favorable per-expert prompt rank as \(E\) grows.

### P7 — V4-Flash flagship

Only after P1–P6:

- target rank 16–32;
- expected mixed atlas 3.36–6.72 GiB;
- target >92.3% fast path for a 10× expert-read reduction;
- final hardware target: 32 GiB RAM + 8 GiB VRAM.

K3 remains the north-star after V4.

---

## 13. Hard falsification rules

Stop RSIV-MoE when any of these is reproduced on V2 and a higher-E second model:

1. At rank 32, double-gate fast-path fraction is below 80% after representative prefill.
2. Rank growth remains approximately linear throughout long decode rather than saturating.
3. The down-intermediate basis requires ranks so high that the image cache loses at least 10× byte advantage.
4. Full-depth CE exceeds 2% despite exact fallback on all registered misses.
5. The physical runtime gains less than 1.5× over packed int4.
6. Cold misses create unacceptable p95 latency even with async prefetch and a warm low-bit tier.

Positive Eureka gate:

> **One frozen model must simultaneously achieve ≥10× routed cold-byte reduction, <2% relative CE loss, ≤8% cold-path invocations, stable 512-token rollouts and ≥2× measured batch-1 decode speed.**

Second-model replication is required before a broad claim.

---

## 14. Final judgment

This is not yet a measured breakthrough. It is the first post-CRAFT hypothesis with all of the following properties:

- mechanistically independent of every CRAFT candidate;
- an exact linear-algebraic core rather than a hoped-for learned surrogate;
- a prompt-exact compilation theorem;
- a model-size scaling law in which the expert count cancels;
- a physically plausible resident-memory budget for V4 and a constrained K3 configuration;
- a clear path to exact/partial fallback;
- cheap, decisive falsification experiments on the already downloaded V2-Lite model.

The essential conceptual change is:

> **Do not compress the model into one smaller static checkpoint. Virtualize it. The full checkpoint is the backing store; the prompt-compiled operator images are the resident executable model.**

