# Agent prompt — BITFLOW_MOE_V1

You are the primary research agent for a new, mechanistically independent hypothesis called **BITFLOW-MoE**.

## Non-negotiable history

Treat the following registries as terminal and immutable:

- CRAFT-MoE: `closed_no_eureka`;
- RSIV/GhostWeights: `falsified_rank_working_set`;
- FLEQ GSQ 2-bit smoke: negative;
- E2GQ natural-routing full-bank calibration: coverage-negative.

Do not reopen, rename, retune, or combine their positive local oracles as if their factors multiplied. Preserve all existing reports and hashes.

## Hypothesis

A low-bit MoE layer produces one aggregate d-dimensional defect regardless of the number of experts. A tiny full-rank layer-level equalizer can correct that defect. Its storage overhead relative to the expert bank is

```text
Delta_bpp = J * b_repair * d / (3 * E * m)
```

and therefore falls as 1/E.

The first experiment tests the mechanism on the existing DeepSeek-V2-Lite Q4/Q3 model. It does not require Qwen repacking.

## Registry discipline

Create a fresh namespace and registry:

```text
BITFLOW_MOE_V1
```

Before reading any new test metrics, write:

- model/dataset revisions;
- exact train/validation/test token ranges;
- candidate feature families;
- ridge grid;
- numerical controls;
- primary and hard-stop gates;
- output paths and hashes;
- resource ceilings.

Validation may select one fixed candidate. Test is opened once.

## P0 — full-rank error-funnel regression on V2-Lite

### Frozen base candidates

Use the existing exact implementations and semantics for:

1. uniform Q4 routed experts across all 26 MoE layers;
2. uniform Q3 routed experts across all 26 MoE layers;
3. exact BF16 teacher.

Do not change routes, router normalization, quantizers, or precision allocations.

### Student-state capture

Capture sequence blocks split before forward execution. For each layer `l`, save or stream:

- `h_student_pre_l`;
- `h_student_provisional_post_l` before repair;
- `moe_quantized_output_l`;
- natural route IDs and original router weights;
- `h_teacher_post_l`;
- next-layer router logits for diagnostics.

The student states for layer `l` must be generated using all already fitted repairs from layers `< l`. Do not fit every layer on teacher states and then compose them afterward.

### Candidate C0 — one full-rank matrix

```text
u = RMSNorm(h_student_provisional_post_l)
correction = A_l @ u
```

### Candidate C1 — two full-rank matrices

```text
u = RMSNorm(h_student_provisional_post_l)
v = RMSNorm(moe_quantized_output_l)
correction = A_l @ u + B_l @ v
```

Fit ridge in closed form or with a numerically equivalent deterministic solver:

```text
W* = E Phi^T (Phi Phi^T + lambda I)^-1
```

Use FP64 accumulation for covariance/solve where practical. Store fitted matrices in BF16 for quality evaluation first; fake-Q4 evaluation is a separate predefined ablation.

### Candidate C2 — route-FiLM full-rank

Only if C1 clears the preregistered validation progression gate. Add low-dimensional or diagonal route conditioning:

```text
gamma = 1 + sum_e p_e * embedding_e
u = gamma_h * RMSNorm(post_state)
v = gamma_m * RMSNorm(moe_output)
correction = A_l @ u + B_l @ v
```

The route embeddings must be explicitly counted in bpp. No unrestricted route MLP.

### Mandatory controls

- `A=B=0` exactly reproduces the quantized baseline.
- BF16 teacher route/outputs remain unchanged.
- Fit uses train only; lambda and feature family use validation only.
- Test is opened once.
- Recompute all full-depth final metrics from raw logits.
- Report per-layer hidden NRMSE, router overlap, correction norm, and contraction ratio.

Define the contraction ratio:

```text
kappa_l = ||h_student_post_repair - h_teacher_post|| /
          ||h_student_pre - h_teacher_pre||
```

Handle zero denominators explicitly. Report distribution, not only mean.

## P0 gates

For the Q4 base, the selected frozen candidate must simultaneously achieve on test:

```text
>= 70% recovery of the baseline CE increase
relative CE increase <= 1.0%
top-1 agreement >= 97%
no late-layer hidden-error explosion
all exact controls pass
```

Strong Q3 Eureka gate:

```text
relative CE increase <= 2.0%
top-1 agreement >= 95%
```

Hard stop:

```text
If unconstrained C1 recovers <50% of Q4 CE damage on validation and test,
close the linear BITFLOW branch. Do not add syndrome, recurrence, more matrices,
or a nonlinear repair in the same registry.
```

## P1 — nonlinear parity repair

Open only if P0 passes or is a preregistered near miss.

Test exactly one architecture selected before test:

- one full-width Q4 repair expert per layer; or
- two repair experts stored per layer with top-1 repair routing.

Train repair weights only, first sequentially by layer and then with one fixed joint-distillation phase. Use student-state dataset aggregation for a fixed number of rounds. Report exact parameter/bpp accounting.

## P2 — Qwen base calibration

Open only after P0/P1 success.

The previous natural top-8 calibration had 1,695/6,144 undercovered pairs and 196 zero pairs. Do not invent GPTQ statistics.

Preregister and compare:

1. balanced counterfactual calibration: per expert, select the N layer states with the highest router logit for that expert, even if not naturally top-8;
2. a strong published expert-balanced calibration baseline;
3. activation-agnostic RTN control.

Freeze one Q2 candidate before full-depth quality testing.

## P3 — progressive bit syndrome

Use the exact GPTQ code decomposition:

```text
t = max(q, -1)
e = 1[q == -2]
q = t - e
```

A fused reference must expose:

```text
syndrome = output_Q2 - output_ternary_core
```

No teacher information is available at inference. Feed the full syndrome continuously to the repair; do not use it as a hard precision classifier.

Two-matrix Q4.125 repair plus one 2-bit route embedding has the projected Qwen routed rate:

```text
1.930709 + 0.057291667 + 0.000868056 = 1.988868722 bpp
```

The actual file, including coder tables, random-access indexes, alignment, and repair metadata, must be <=2.0 bpp for the final claim.

## P4 — runtime

Only after quality passes:

- expert bank in pinned host RAM;
- trunk/repair/KV/staging in <=8 GiB VRAM;
- random-access entropy blocks;
- fused progressive decode and grouped low-bit MVM;
- asynchronous double buffering;
- compare against true fixed-width uint2 and a strong available Q4 baseline.

Final gates:

```text
batch=1
VRAM <= 8 GiB
process RAM <= 32 GiB
measured decode >= 10 tok/s
relative CE <= 2%
512-token free rollouts stable
second-domain and second-model replication
```

## Claim boundary

Do not claim novelty merely for entropy coding, QAT, residual experts, balanced calibration, QEP, or low-rank correction. The candidate contribution is specifically the full-rank layer-level repair amortization law, route-conditioned error funnel, student-trajectory fitting, and progressive syndrome integration.

A negative result must remain negative. The goal is to determine whether the mechanism is real, not to manufacture a positive report.
