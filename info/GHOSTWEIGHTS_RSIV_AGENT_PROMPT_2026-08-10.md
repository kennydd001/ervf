# Agent prompt — RSIV-MoE / GhostWeights

You are the principal research engineer for a new, mechanistically independent inference project named **RSIV-MoE (Routed Subspace Image Virtualization)**, product codename **GhostWeights**.

The previous CRAFT-MoE registry is frozen as `closed_no_eureka`. Do not reopen, retune or reinterpret any CRAFT hypothesis. Create a new registry namespace and preserve all existing files.

## Objective

Test whether a disk-backed MoE can keep only prompt/workload-specific operator images resident, so expert inference cost scales with routed activation rank rather than total expert parameters.

The target platform is:

- Windows 11 / optional WSL or native Linux where required;
- 64 GiB physical RAM, but report a constrained 32 GiB process budget;
- RTX PRO 2000 Blackwell Laptop GPU with 8 GiB VRAM;
- existing pinned `deepseek-ai/DeepSeek-V2-Lite` Base checkpoint and traces.

Do not download DeepSeek-V4-Flash or Kimi K3 during the first phase.

## Core construction

For each routed SwiGLU expert:

```text
g = G x
u = U x
z = SiLU(g) * u
y = D z
```

Maintain an input basis `Q_e` and an intermediate basis `P_e`, plus images:

```text
A_e = G_e @ Q_e
B_e = U_e @ Q_e
C_e = D_e @ P_e
```

Fast path:

```text
q = Q_e.T @ x
g_hat = A_e @ q
u_hat = B_e @ q
z_hat = silu(g_hat) * u_hat
p = P_e.T @ z_hat
y_hat = C_e @ p
```

Gates:

```text
rho_x = ||x - Q_e Q_e.T x|| / ||x||
rho_z = ||z_hat - P_e P_e.T z_hat|| / ||z_hat||
```

On a gate miss, execute only the required original projection(s), append the normalized residual direction to the relevant basis, and append the corresponding exact operator-image column. The full weights are a cold backing store, not resident state.

## Mandatory mathematical controls

1. Prompt inputs routed to expert `e` form `X_e`; intermediates form `Z_e`.
2. Verify `rank(X_e) <= n_e`, `rank(Z_e) <= n_e`.
3. Verify `sum_e n_e = top_k * T` per layer.
4. Verify the bound:

```text
sum_e [(d + 2m) r_e + (m + d) s_e] <= (2d + 3m) top_k T
```

5. At full-rank prompt compilation, reproduce every calibration/prompt expert output within a preregistered numerical tolerance.
6. Original/full fallback control must remain exact.

## Phase 1 — Rank census before building a runtime

Create:

```text
reports/rsiv_moe/RSIV_MOE_PREREGISTRATION.md
reports/rsiv_moe/EXPERIMENT_REGISTRY.yaml
scripts/rsiv_moe/measure_routed_subspace_rank.py
src/moe_lab/rsiv_moe/subspace.py
tests/rsiv_moe/test_subspace.py
```

Measure on layers 1, 13 and 26 first, then all 26 layers if controls pass.

For each expert and split:

- routed input count;
- exact rank;
- effective rank;
- causal online rank-growth curve;
- held-out residual ratio at ranks `4,8,16,32,64,128`;
- prompt-prefix to future-token fast-path transfer;
- router-mass-weighted and invocation-weighted statistics;
- rare-expert coverage.

Run both:

- offline train/calibration basis;
- causal prefix-only basis, tested only on later positions.

Threshold grid:

```text
0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10
```

No test-selected rank or threshold. Choose on validation only and open test once.

Primary screen:

```text
rank_cap <= 32
double_gate_fast_fraction >= 0.92
projected_routed_cold_byte_reduction >= 10x
```

Do not terminate the entire research line solely because V2-Lite misses this screen: the new scaling law predicts more favorable average observations per expert for higher-E models. A V2 failure must still be reported honestly.

## Phase 2 — Real operator-image oracle

Build actual `A/B/C` images from original weights. Evaluate:

- exact prompt reconstruction;
- held-out one-layer output error;
- gate/up miss rate;
- down miss rate;
- BF16, FP8 and INT8 image formats;
- image bytes versus packed-int4 full-expert bytes;
- compute and transfer accounting.

Mandatory baselines:

- full BF16;
- packed/fake int4 expert;
- static per-layer GSI-style basis;
- per-expert input basis only, full down;
- per-expert input and intermediate bases;
- no-gate static projection as a negative control.

## Phase 3 — Full-depth causal simulator

Every candidate follows its own hidden states and routes through all 26 MoE layers.

On each miss:

- compute the original projection exactly;
- update the basis and image cache causally;
- count full/partial expert bytes;
- record rank growth.

Report:

- final KL;
- relative CE;
- top-1 agreement;
- per-layer router overlap;
- fast/slow path fraction;
- gate/up-only and down-only misses;
- image cache growth;
- cold bytes/token p50/p95/p99;
- estimated and measured latency when available.

Primary gates:

```text
relative_CE_delta < 0.02
routed_cold_byte_reduction >= 10x
slow_path_invocation_fraction <= 0.08
exact_control == pass
```

Secondary tail gate:

```text
slow_path_invocation_fraction <= 0.03
```

## Phase 4 — Three-tier residual path

Only after the basic operator-image path is measured, test:

```text
y_hat = (WQ)(Q.T x) + Q_lowbit(W) r
```

with optional full-weight fallback for large residuals.

Use 1-, 2- and 3-bit warm residual operators. Report the exact bound proxy:

```text
||W - Q_lowbit(W)||_2 * ||r||
```

and actual downstream metrics. Do not compare only local MSE.

## Phase 5 — Prompt compilation and rollouts

At least 20 prompts and 512 generated tokens per prompt across code, reasoning, Dutch, ordinary English and retrieval.

Measure:

- prefill compilation cost;
- rank at end of prefill;
- rank additions per generated token;
- atlas reuse across turns;
- p95/p99 cold faults;
- quality/task metrics;
- stable autoregression.

## Phase 6 — Packed runtime

Only if the oracle gates pass:

- mixed BF16 basis + INT8/FP8 image layout;
- partial gate/up/down page faults;
- pinned host buffers;
- asynchronous NVMe prefetch;
- byte-budgeted image cache;
- real packed-int4 baseline;
- batch-1 tokens/s and energy.

Runtime Eureka gate:

```text
>= 2x measured decode speed
< 2% relative CE loss
>= 10x routed cold-byte reduction
no unacceptable p95 latency spikes
```

## Phase 7 — Generalization

Replicate on one higher-expert-count family, preferably Qwen3-30B-A3B or Kimi-Linear-48B-A3B. Only then consider V4-Flash.

The method's central predicted scaling quantity is:

```text
average_prompt_inputs_per_expert = top_k * prompt_tokens / expert_count
```

Report whether per-expert ranks follow this favorable scaling.

## Research conduct

- Preregister every primary gate before opening test data.
- Preserve failed runs and raw artifacts.
- Do not claim speed from byte accounting.
- Do not multiply factors from incompatible interventions.
- Distinguish exact algebra, empirical measurement, projection and assumption.
- Run primary-literature and limited patent searches before making novelty claims.
- Treat Gated Subspace Inference as an unverified recent preprint until independently reproduced.
- Return a terminal verdict: `falsified`, `inconclusive`, `engineering_positive`, or `eureka_confirmed`.

## Final Eureka definition

A single frozen candidate must jointly demonstrate:

```text
>= 10x routed cold-byte reduction
< 2% relative CE loss
<= 8% cold-path invocations
stable 512-token rollouts
>= 2x measured batch-1 decode speed
replication on a second MoE family
```

Anything less may be useful engineering, but is not the requested breakthrough.
