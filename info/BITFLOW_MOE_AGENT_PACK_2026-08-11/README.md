# BITFLOW-MoE: full-rank error-funnel repair for entropy-coded experts

**Date:** 2026-08-11  
**Status:** new mechanistically independent Eureka hypothesis; not yet an empirical breakthrough  
**Proposed registry ID:** `BITFLOW_MOE_V1`

## Executive verdict

The combined CRAFT, RSIV, FLEQ, E2GQ, and HERA evidence rules out a broad family of attractive but insufficient mechanisms:

- the routed expert function is not usefully represented by one small shared output subspace;
- low-rank activation working sets do not emerge after long prefill on either DeepSeek-V2-Lite or Qwen3-30B-A3B;
- local route/precision and atom-sparsity oracles do not stack safely through all layers;
- hard dynamic-precision selectors miss too many high-damage events;
- natural co-routed quantization-error cancellation is essentially absent;
- a 2-bit GSQ pass did not improve the locked GPTQ baseline on any of the sixteen tested Qwen experts;
- nevertheless, the locked GPTQ code stream has real entropy below 2 bits/weight, and the remaining deployment bandwidth is physically plausible.

The common failure is not lack of local redundancy. The failure is **full-rank, cumulative state error**. The new proposal therefore makes no low-rank assumption and does not try to repair every expert separately.

> **BITFLOW-MoE stores a full-rank low-entropy expert bank and adds a tiny, always-on, route-conditioned full-rank error equalizer at the one place where all expert errors must pass: the d-dimensional residual stream of each layer.**

The key scaling result is that a full-rank correction is surprisingly cheap in a large MoE because its cost is amortized over all experts in the layer.

---

## 1. Evidence synthesis

### 1.1 What CRAFT established

CRAFT's strongest local signals were not stackable. The local Q3/Q4 route oracle, exact atom oracle, block coalescing, error sketches, cache-span reconstruction, and co-routed cancellation all failed their required downstream, full-depth, or system gates. Mass-Budget remained a valid incremental cache baseline, not an order-of-magnitude mechanism.

The important mechanistic result is that errors which are locally small can be amplified or redirected by later layers. A model-wide repair must therefore operate on student states and must be trained against the accumulated trajectory, not only teacher-forced local outputs.

### 1.2 What RSIV established

RSIV/GhostWeights showed that routed input and SwiGLU-intermediate activations are essentially full observation-rank. On Qwen3-30B-A3B, rank 32 reached only 1.742% double-fast and 1.034x cold-byte reduction; rank 128 reached only 5.762% and 1.108x. The future working set does not saturate into a useful low-rank atlas.

This kills low-rank operator virtualization, but it does **not** imply that the weights or their error must require high entropy.

### 1.3 What FLEQ and E2GQ established

The official GSQ operator was tested expert-by-expert on a locked Qwen sample. The 2-bit GSQ candidate improved zero of sixteen experts over GPTQ, and all sixteen p95 errors regressed. Ternary GSQ beat hard ternary RTN but remained substantially worse than 2-bit GPTQ.

The same locked GPTQ artifacts yielded a positive representation result:

- code histogram over 75,497,472 weights:
  `{-2: 4,713,974; -1: 17,846,753; 0: 31,599,966; +1: 21,336,779}`;
- zero-order code entropy: `1.782864891374` bpp;
- raw BF16 group-128 scales: `0.125` bpp;
- ideal total: `1.907864891374` bpp;
- exploratory zlib-9 physical projection: `1.930709` bpp.

The full-bank natural-routing calibration then failed coverage: 1,695 of 6,144 layer-expert pairs had fewer than 128 routed examples and 196 had zero. This falsifies that calibration rule, not low-entropy representation itself.

### 1.4 What HERA did and did not establish

HERA proposed an entropy-coded hot tier and exact cold tier. Its first multidomain attempt had invalid route-ID controls because a second BF16 `topk` call changed tied indices. Therefore the reported 6,081/63 hot/cold union is not a valid final result. HERA remains unresolved and cannot be used as proof for or against static tiering.

---

## 2. The error-funnel observation

For layer `l`, let the full-precision routed MoE contribution be

\[
m_l^T(h)=\sum_{e\in R_l(h)}p_{l,e}(h)E_{l,e}^T(h)\in\mathbb R^d
\]

and the low-bit contribution be

\[
m_l^Q(h)=\sum_{e\in R_l(h)}p_{l,e}(h)E_{l,e}^Q(h).
\]

The aggregate quantization defect is

\[
\epsilon_l(h)=m_l^T(h)-m_l^Q(h)\in\mathbb R^d.
\]

Regardless of whether the layer contains 64, 128, 256, or 896 experts, all selected-expert errors are forced through a **single d-dimensional additive bottleneck** before the next layer. The error may be full-rank across data; it is still one layer-level function, not `E` independent deployed functions.

The proposed repaired student is

\[
\bar h_{l+1}^Q = F_l^Q(h_l^S),
\]

\[
h_{l+1}^S = \bar h_{l+1}^Q + C_l(\text{observable sensors at layer }l).
\]

The repair is trained on student trajectories, with target

\[
c_l^*=h_{l+1}^T-\bar h_{l+1}^Q.
\]

It does not reconstruct individual experts and it does not require the error to be low-rank.

---

## 3. Error-Funnel Amortization Law

A standard SwiGLU expert contains approximately

\[
3dm
\]

weights. A layer with `E` experts contains

\[
3Edm
\]

routed-expert weights.

Add `J` full-rank `d x d` repair matrices per layer, stored at effective precision `b_r`. Their equivalent overhead, measured in bits per original routed-expert weight, is

\[
\boxed{\Delta b = Jb_r\frac{d}{3Em}}.
\]

The active repair compute relative to `k` active standard experts is

\[
\boxed{\Delta C = J\frac{d}{3km}}.
\]

For fixed architecture ratios, repair storage decreases as `1/E`. This is the opposite scaling behavior of per-expert residuals.

### Qwen3-30B-A3B example

Use the pinned architecture from the supplied RSIV registry:

- `L=48`, `E=128`, `k=8`, `d=2048`, `m=768`;
- routed expert parameters: `28,991,029,248`;
- non-expert parameters inferred from the exact BF16 checkpoint: `1,542,258,576`.

For two full-rank repair matrices per layer at Q4 plus raw BF16 group-128 scales (`4.125` effective bits per repair weight):

\[
\Delta b = 2\cdot4.125\frac{2048}{3\cdot128\cdot768}
         = 0.057291667\text{ bpp}.
\]

One 2-bit route embedding per expert adds

\[
\frac{2}{3m}=0.000868056\text{ bpp}.
\]

Using the supplied exploratory physical GPTQ coder rate of `1.930709` bpp:

\[
1.930709+0.057291667+0.000868056
=\boxed{1.988868722\text{ bpp}}.
\]

Thus the measured entropy reserve can theoretically finance **two full-rank correction channels per layer** while remaining below 2 bpp for the routed-bank representation.

### Projected Qwen storage

| Component | Projected size |
|---|---:|
| Expert bank at 1.930709 bpp | 6.516 GiB |
| Two Q4.125 full-rank repair matrices/layer | 0.193 GiB |
| One 2-bit route embedding/expert/layer | 0.003 GiB |
| Non-expert trunk at ideal INT4 | 0.718 GiB |
| Total weight projection | **7.431 GiB** |

This is too tight for an all-VRAM deployment after KV cache and buffers. The intended hierarchy is instead:

- entropy-coded experts in host RAM;
- trunk, repair matrices, KV cache, and staging buffers in VRAM;
- only the eight active expert chunks streamed each layer.

Projected active expert traffic from the supplied physical coder is about `416.438 MiB/token`. Repair matrices remain resident in VRAM. At 10 tokens/s, host-to-device expert traffic is about `4.07 GiB/s`, before headers, alignment, decoder overhead, and prefetch inefficiency.

This is a feasibility projection, not a speed result.

---

## 4. BITFLOW architecture

### 4.1 Progressive low-bit sensor

The measured GPTQ alphabet is `{-2,-1,0,+1}`. It has an exact decomposition

\[
t=\max(q,-1),\qquad e=\mathbf1[q=-2],\qquad q=t-e.
\]

The entropy pack can therefore expose two exact streams:

- a ternary core;
- a sparse extreme-tail stream.

A fused kernel can accumulate both the core output and the tail contribution from the same compressed code stream. Define the tail contribution to the MoE output as

\[
s_l = m_l^{Q2}-m_l^{\text{ternary-core}}.
\]

` s_l ` is a directional, d-dimensional quantization syndrome. It requires no teacher and no new weight stream. The tail occupies only 6.244% of the locked codes, although full-bank transfer must be re-measured.

CRAFT's random residual sketch was unsafe as a hard binary precision selector, but it still recovered over 80% of oracle KL benefit. BITFLOW does not use a lossy sketch as a gate. It gives the full progressive syndrome to a continuous full-rank corrector.

### 4.2 Route-conditioned full-rank equalizer

A compact two-channel version is

\[
u_l^h = \gamma_l^h(R_l,p_l)\odot\operatorname{RMSNorm}(\bar h_{l+1}^Q),
\]

\[
u_l^s = \gamma_l^s(R_l,p_l)\odot\operatorname{RMSNorm}(s_l),
\]

\[
\boxed{c_l=A_lu_l^h+B_lu_l^s},
\]

where `A_l,B_l in R^{d x d}` are full-rank repair matrices and the route-conditioned diagonal gates are generated from low-bit per-expert embeddings:

\[
\gamma_l(R,p)=1+\sum_{e\in R}p_e v_{l,e}.
\]

The corrected state is

\[
h_{l+1}^S=\bar h_{l+1}^Q+c_l.
\]

Because `A_l` and `B_l` are full-rank, this architecture is not contradicted by the high behavioral rank or by RSIV's full-rank activation census.

### 4.3 Flow across layers

The corrected student state itself transports the repair downstream. A later extension may expose the previous correction as another diagonally gated sensor:

\[
u_l^h \leftarrow u_l^h + \eta_l\odot c_{l-1}.
\]

This creates a learned error-flow observer without another dense matrix. It must not be added unless the simpler preregistered equalizer is a near miss.

---

## 5. Why this is not a renamed failed CRAFT method

| Closed method | BITFLOW difference |
|---|---|
| Shared output basis / low-rank repair | repair is explicitly full-rank in the residual stream |
| Aggregate surrogate | low-bit experts still perform the main function; repair learns only their aggregate defect |
| QERC error cancellation | no assumption of natural cancellation between expert errors |
| SketchGate | syndrome is a continuous correction input, not a hard safety classifier |
| CRCQ rerouting | natural route can remain unchanged |
| Atomic sparsity | no static neuron removal is required |
| RSIV | no low-rank activation or future working-set assumption |
| Per-expert low-rank residual | repair cost is layer-level and amortized over all experts |

---

## 6. Training protocol: closed-loop, not teacher-forced only

A frozen low-bit base and repair-only training are sufficient for the first test.

### Sequential layer fit

For each layer `l`:

1. Generate teacher state `h_l^T` and current repaired-student state `h_l^S` on train sequences.
2. Run the frozen low-bit layer to obtain `bar h_{l+1}^Q` and all inference-visible sensors.
3. Fit the repair to
   \[
   c_l^*=h_{l+1}^T-\bar h_{l+1}^Q.
   \]
4. Freeze the fitted repair temporarily.
5. Regenerate downstream student states before fitting layer `l+1`.

This directly accounts for accumulated error and router drift.

### Joint refinement

After the sequential pass, jointly optimize repair parameters with

\[
\mathcal L =
\lambda_h\sum_l\|h_l^T-h_l^S\|_2^2
+\lambda_r\sum_lD_{KL}(r_l^T\|r_l^S)
+\lambda_zD_{KL}(p_T\|p_S)
+\lambda_c\sum_l\|c_l\|_2^2.
\]

### On-policy dataset aggregation

For autoregressive stability:

1. generate continuations with the repaired student;
2. query the frozen teacher on those student prefixes;
3. add those states to the repair dataset;
4. repeat for a fixed number of preregistered rounds.

No teacher or extra model is needed at deployment.

---

## 7. Cheapest decisive experiment

Do **not** begin with Qwen full-bank 2-bit packing. The existing DeepSeek-V2-Lite full-depth Q4/Q3 infrastructure can test the error-funnel mechanism cheaply.

### P0 — full-rank linear oracle on V2-Lite

Use the frozen, existing uniform-Q4 and uniform-Q3 student implementations.

For each of the 26 MoE layers, capture on train only:

- current student pre-layer state;
- provisional quantized post-layer state;
- quantized routed MoE output;
- route IDs and original router weights;
- teacher post-layer state.

Fit and compare, with validation-only ridge selection:

1. `A * post_quant_state`;
2. `A * post_quant_state + B * quantized_moe_output`;
3. route-FiLM plus the same two full-rank matrices;
4. the previously tested low-rank correction as a mandatory control.

Fit sequentially on regenerated student states. Open test once.

### P0 success gates

For uniform Q4:

- recover at least 70% of the baseline CE increase;
- final relative CE increase <=1.0%;
- final top-1 agreement >=97%;
- no late-layer hidden-error explosion.

For uniform Q3, a strong-Eureka gate is:

- relative CE increase <=2%;
- final top-1 agreement >=95%.

A P0 failure is terminal for the linear equalizer, not automatically for a nonlinear parity expert. However, if the unconstrained two-matrix full-rank regression recovers less than 50% of Q4 CE damage, do not build the bit-syndrome runtime.

### P1 — nonlinear parity repair, only after P0

Replace the linear equalizer with one of the following, fixed before test:

- one Q4 full-width repair expert per layer; or
- two Q4 half/full-width repair experts with top-1 repair routing.

The total routed-equivalent rate must remain within the registered budget.

### P2 — Qwen low-bit base

Before Qwen quality testing, solve expert calibration with a separately preregistered mechanism. The primary candidate is balanced counterfactual calibration:

- collect layer hidden states once;
- for each expert, select the `N` states with the largest router logit for that expert, even when it was not in natural top-8;
- calibrate the expert only on that router-near manifold;
- compare against EAQuant/MoEQuant-style balanced calibration and activation-agnostic RTN.

This avoids silently inventing statistics for zero-coverage experts.

### P3 — Qwen BITFLOW

Freeze one entropy-coded Q2 base and one repair architecture. Require:

- actual routed-bank file <=2.0 bpp including scales, indexes, coder tables, and alignment;
- full-depth relative CE increase <=2%;
- stable 512-token rollouts on general text, code, math, multilingual, and instructions;
- no catastrophic expert/domain tail.

### P4 — physical runtime

- compressed experts in pinned host RAM;
- trunk and repair resident in 8 GiB VRAM;
- random-access block entropy coding;
- fused progressive core/tail decode plus grouped low-bit MVM;
- asynchronous double-buffered H2D transfer.

Final gate:

```text
batch=1
VRAM <= 8 GiB
process RAM <= 32 GiB
measured decode >= 10 tok/s
relative CE <= 2%
512-token rollouts stable
```

---

## 8. Novelty boundary

The broad ingredients are prior art:

- low-bit MoE quantization and compressed-domain kernels;
- entropy coding;
- expert-balanced calibration;
- quantization-error propagation;
- low-rank quantization compensation;
- residual expert architectures.

The targeted literature search did not locate the exact conjunction below:

> A frozen low-entropy MoE expert bank plus a route-conditioned **full-rank layer-level repair equalizer**, whose storage overhead vanishes as `O(1/E)`, trained sequentially on student trajectories and optionally driven by the exact progressive low-bit tail syndrome.

That is negative search evidence, not proof of novelty or patentability.

Closest required baselines include QEP, RILQ, EAQuant, Preserve-Then-Quantize, QMoE, GSQ, and residual-expert architectures. BITFLOW must beat them on a matched total-rate and actual-runtime basis.

---

## 9. Risks and falsification

The hypothesis can fail for several reasons:

1. Quantization error may be full-rank **and** too nonlinear/route-specific for one or two shared full-rank maps.
2. Sequential repair may overfit calibration and fail autoregressive states.
3. A Q2 base may be so damaged that the repair becomes a second model rather than a correction.
4. Random-access entropy coding may lose the measured rate advantage.
5. Host-to-device transfer and entropy decoding may dominate wall-clock time.
6. Balanced counterfactual calibration may not approximate each expert's natural hidden-state distribution.

Hard stop:

- if the unconstrained V2 full-rank P0 recovers <50% of Q4 CE damage;
- or if a repaired Qwen candidate cannot stay below 2% relative CE at <=2 bpp;
- or if the physical implementation remains below 5 tok/s after kernel profiling.

No post-hoc architecture sweep should reopen a failed registered gate.

---

## 10. Secondary hypothesis if BITFLOW fails

A more radical, higher-cost direction is to retrain experts as full-rank structured orbits:

\[
W_e \approx \sum_{j=1}^{J} U_j\operatorname{diag}(a_{e,j})V_j,
\]

with shared fast orthogonal/butterfly transforms and expert-specific diagonal spectra. This preserves full rank while changing parameter scaling from independent matrices to shared transforms plus per-expert diagonals. ButterflyMoE-like work is close prior art, and this route requires substantial continued pretraining. It is Plan B, not a reason to weaken BITFLOW's gates.

---

## Final research claim

No Eureka has yet been measured. The new scientific candidate is:

> **High-rank model error need not require high-rate per-expert repair. In a large MoE, all expert quantization defects funnel through one residual vector per layer, so one full-rank error equalizer can be amortized over the entire expert bank at vanishing bits per original expert weight. The entropy reserve measured in Qwen's existing GPTQ codes is large enough to finance that equalizer without exceeding approximately 2 bpp.**

This is the first new mechanism in the bundle that is simultaneously:

- compatible with the CRAFT and RSIV negative results;
- mathematically full-rank;
- explicitly designed for accumulated student-state error;
- small enough to fit the measured entropy budget;
- and capable, if the quality oracle succeeds, of making 10 tok/s physically plausible on the target memory hierarchy.
