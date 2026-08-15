# N6-A — full-depth forward, natural routes and coherence

Datum: 2026-08-14
Verdict: **PASS. The assembled 52-layer graph produces coherent language on real weights. `The capital of France is` → ` Paris`, p = 0.9603.**
Terminal state: `n6a_full_depth_coherent`
Independent verification: **32/32**

## Kernresultaat

| id | prompt | top-1 | p | entropy | s |
|---|---|---|---:|---:|---:|
| **P1** | `The capital of France is` | **` Paris`** | **0.9603** | 0.248 | 90.2 |
| P2 | `1, 2, 3, 4,` | `' '` | 0.9255 | 0.451 | 112.9 |
| P3 | `def add(a, b):\n    return` | **` a`** | 0.9865 | 0.125 | 113.8 |

P1 is the gated prompt and it passes. P3 is corroborating and is exactly right —
` a` is the correct continuation of `return` in `def add(a, b)`. P2 selects a
space, which is a tokenisation artefact of digit sequences rather than a failure;
it was preregistered as reported-not-gated for precisely this reason.

All 52 layers executed for every prompt: 23 Mamba-2, 23 MoE, 6 attention,
following the frozen `hybrid_override_pattern`.

## Wat dit afsluit

Coherence is a **joint** test. Several assumptions in this line pass every local
check and would still yield a wrong model; a 52-layer forward that emits the
expected token exercises all of them simultaneously.

| assumption | status before N6-A | status now |
|---|---|---|
| nibble order `low_first` | confirmed against torchao only | jointly confirmed in a running model |
| **gated RMSNorm semantics** | **unvalidated** — `mamba_ssm` needs CUDA, so our implementation was used on both sides of the N3 comparison | **jointly confirmed** |
| dequant grouping, `input_scale` role | structurally confirmed | jointly confirmed |
| `up → ReLU² → down`, ungated shared expert, router bias split | validated per module | confirmed in composition |

This settles the debt N3 recorded explicitly and deferred to "end-to-end
coherence at N6". It is now paid.

The honest limit of that statement is recorded in the result itself: passing
supports the assumption set **as a whole**; a failure would not have identified
which member was wrong. Diagnosis would have been a separate phase.

## Natuurlijke routes

Every prior phase used the frozen N3 capture, which is a *synthetic-input* route
set. N6-A produces the first **natural** routes — from a real forward on real
text — for all 23 MoE layers.

| quantity | value |
|---|---:|
| route rows captured | 552 |
| MoE layers per prompt | 23 |
| experts per token | 6, all unique within a token |
| route weights | finite, positive, summing to 2.5 per token |
| **distinct experts used** | **128 / 128 (100.0%)** |
| usage max / min | 61 / 7 |

**All 128 experts of every layer appear.** At this sample size there is no small
hot set: the most-used expert is selected 61 times and the least-used 7, a ratio
of 8.7 across 552 rows. That is a direct input to H5 — a cache sized at the 572
slots N5 measured cannot be filled by "the popular experts", because at this
sample the popularity spread is shallow.

That observation is bounded by the sample. Three prompts and 552 rows are not a
routing distribution, and the artifact says so in its own provenance field.

## Meetprotocol

- CPU only, float64 accumulation, N3-validated numpy modules. **No GPU**, so the device stayed free for the protected line.
- **No timing claim.** The per-prompt seconds are recorded for reproduction cost only; a numpy CPU reference is not a runtime.
- Weights dequantised **per layer and released**. The full model in float32 would be roughly 117 GB; peak process commit was **6.734 GiB** against a 32 GiB gate.
- Prompts and the expected P1 continuation were frozen in the preregistration before execution.

One implementation fact worth recording: the first run failed on
`backbone.layers.4.mixer.in_proj.weight_scale`. That tensor does not exist —
layers 4, 11, 18, 25, 32 and 41 keep their Mamba `in_proj`/`out_proj` in BF16,
exactly as the `exclude_modules` analysis in N0R predicted. The fix was to load
through the loader's dtype-agnostic path. The failure is therefore a
**confirmation** of the N0R reading, not a contradiction of it.

## Onafhankelijke verificatie

A separate verifier re-tokenised the frozen prompts, re-decoded the recorded
top-1 ids through the tokenizer, and re-audited the natural-route capture from
the raw arrays — layer coverage, id validity, uniqueness within a token, weight
sums against `routed_scaling_factor`, tie-margin finiteness and usage
statistics.

It deliberately does **not** re-run the forward: that would re-execute the same
code rather than check it, at ~90 s per prompt. What it verifies is that the
artifacts are internally consistent, that the coherence claim rests on a
correctly decoded token, and that the route capture downstream phases will
consume is well formed.

Result: **32/32 verification checks passed.**

## Gates

| # | gate | result |
|---|---|:--:|
| C1 | all prompts completed 52 layers | ✅ |
| C2 | all logits and hidden states finite | ✅ |
| C3 | natural routes for all 23 MoE layers | ✅ |
| C4 | route weights valid and correctly scaled | ✅ |
| C5 | **P1 top-1 decodes to `Paris`** | ✅ |
| C6 | distribution not degenerate | ✅ |
| C7 | process commit ≤ 32 GiB | ✅ 6.734 |
| C8 | no protected byte changed | ✅ |

## Eerlijk verdict

What N6-A establishes: the complete 52-layer graph, assembled from independently
validated modules and running on the real NVFP4 checkpoint, produces correct and
confident next-token predictions on frozen prompts, and yields natural routes for
every MoE layer. The last unvalidated numerical assumption in the line is closed.

What N6-A does **not** establish: model quality, any benchmark score, tokens per
second, latency, correctness in general, or a representative routing
distribution. Three prompts are three prompts. The forward is a numpy CPU
reference, not the runtime — the runtime pieces measured in N4-R2 and N5 are a
separate construction that this phase does not exercise.

The two halves of this project now exist separately and both work: a **correct**
full-depth graph (N6-A, CPU) and a **fast, memory-feasible** routed dataplane
(N4-R2 + N5, GPU). Joining them is the next construction, and it is engineering
rather than research — every component question it depends on has been answered.

## Vervolg

`N6-B`: the GPU decode loop. Assemble the N5 resident shell plus the N4-R2
overlapped routed path into a single-token forward on device, gated on
reproducing this phase's logits within a declared tolerance and on the same
natural routes.

Only after that does H5 cache work become meaningful — and N6-A has already
sharpened it: with 128/128 experts touched and a shallow popularity spread, a
static-prior cache is unlikely to earn its slots, so the validation-selected
static plus causal-dynamic split from STREAMQ5 must be re-derived here rather
than inherited.

## Artefacten

- Preregistratie: `reports/lightningstream_nemotron/N6A_FULL_DEPTH_FORWARD_PREREGISTRATION_2026-08-14.md`
- Runner: `scripts/lightningstream_nemotron/n6a_full_depth_forward.py`
- Machine-readable result: `reports/lightningstream_nemotron/n6a_full_depth_forward.json`
- Natural routes: `reports/lightningstream_nemotron/n6a_natural_routes.json`
- Independent verifier: `scripts/lightningstream_nemotron/n6a_independent_verify.py`
- Verification output: `reports/lightningstream_nemotron/n6a_independent_verification.json`
