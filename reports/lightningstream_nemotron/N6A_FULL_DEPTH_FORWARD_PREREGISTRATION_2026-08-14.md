# N6-A — full-depth forward, natural routes and coherence

Registry: LIGHTNINGSTREAM_NEMOTRON · Phase: `N6_A_FULL_DEPTH_FORWARD`
Datum: 2026-08-14
Status bij schrijven: **design frozen, execution not yet authorized**
Depends on: `N5_PHYSICAL_RESIDENT_SHELL` (PASS, 32/32)
Protected baseline: root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`

## 1. Vraag

Does the complete 52-layer graph, assembled from the modules N3 validated
individually, produce **coherent language** — and what are the **natural routes**?

Two debts are settled here:

1. **The gated RMSNorm gap.** N3 could not validate it: `mamba_ssm` needs CUDA
   and cannot be installed, so our implementation was used on both sides of that
   comparison. N3 explicitly deferred it to "end-to-end coherence".
2. **Synthetic routes.** Every phase so far used the frozen N3 capture, which is
   a *synthetic-input* route set. Nothing downstream — cache design above all —
   can be honest without natural routes from a real forward.

## 2. Why coherence is the decisive test

Several assumptions in this line pass every local check and would still produce
a wrong model:

| assumption | local status | what a wrong value looks like |
|---|---|---|
| nibble order `low_first` | confirmed vs torchao | scrambled weights, fluent-shaped garbage |
| gated RMSNorm semantics | **unvalidated** | wrong Mamba output scale, drifting text |
| dequant grouping / `input_scale` role | structurally confirmed | systematically mis-scaled experts |
| `up → ReLU²  → down` ordering | validated vs official | wrong expert output |

None of these is detectable from a single module in isolation once both sides
share the same code. A 52-layer forward that emits the expected next token is a
**joint** test of all of them at once. That is why it is worth running even
though it produces no performance number.

## 3. Method

CPU, float64 accumulation, using the N3-validated numpy reference modules.
Weights are dequantised **on demand per layer and released**, so no full BF16
model is materialised at any point — the whole model in float32 would be
roughly 117 GB and the 32 GiB process gate forbids it.

The runner walks the frozen `hybrid_override_pattern`
`MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`, executing for each
layer `residual + mixer(RMSNorm(h))`, then the final norm and the LM head.

No GPU is used. This phase makes **no timing claim**, so a CPU run is adequate
and keeps the device free for the protected line.

## 4. Prompts

Three frozen prompts, chosen so the expected continuation is unambiguous and
would be destroyed by any of the §2 failures:

| id | prompt | expected property |
|---|---|---|
| P1 | `The capital of France is` | top-1 should be the ` Paris` token |
| P2 | `1, 2, 3, 4,` | top-1 should continue the counting sequence |
| P3 | `def add(a, b):\n    return` | top-1 should be plausible Python continuation |

P1 is the primary gate. P2 and P3 are corroborating and are reported but not
gated, because tokenisation of digits and code is model-specific and a
reasonable model may legitimately choose several continuations.

**Declared in advance:** P1 passing is evidence the joint assumption set is
right; P1 failing is evidence something in §2 is wrong and does **not** by
itself identify which. Diagnosis would be a separate phase.

## 5. Frozen gates

| # | gate | threshold |
|---|---|---|
| C1 | forward completes all 52 layers for every prompt | required |
| C2 | all hidden states and logits finite | required |
| C3 | natural routes captured for all 23 MoE layers, exactly 6 per layer, every id in `[0, 128)` | required |
| C4 | route weights finite, positive, and summing to `routed_scaling_factor` per token before scaling | required |
| C5 | **P1 top-1 token decodes to a string containing `Paris`** | **required** |
| C6 | logits are not degenerate (top-1 probability < 0.999 and at least 100 distinct top-1 across vocab positions is not required; simply that the distribution is not a constant) | required |
| C7 | no full BF16 model materialised; process peak commit ≤ 32 GiB | required |
| C8 | no protected byte changed | required |

C5 is the phase. If it fails, the terminal state is
`n6a_incoherent_assumption_set_wrong` and the line stops to diagnose rather than
proceeding to a cache.

## 6. Claim boundary

N6-A may claim only: that the assembled 52-layer graph runs to completion on
real weights and produces a specific next-token prediction for specific frozen
prompts, and the natural routes observed for those prompts. It may **not** claim
model quality, benchmark scores, tokens per second, latency, that the runtime is
correct in general, or that these routes generalise — three prompts are three
prompts.

The captured routes are **natural** in the sense that they come from a real
forward on real text, and must still be described as coming from these three
prompts only, not as a representative routing distribution.

## 7. Artefacten

| path | kind |
|---|---|
| `scripts/lightningstream_nemotron/n6a_full_depth_forward.py` | runner |
| `scripts/lightningstream_nemotron/n6a_independent_verify.py` | independent verifier |
| `reports/lightningstream_nemotron/n6a_full_depth_forward.json` | raw result |
| `reports/lightningstream_nemotron/n6a_natural_routes.json` | natural route capture |
| `reports/lightningstream_nemotron/n6a_independent_verification.json` | verifier output |
| `reports/lightningstream_nemotron/N6A_FULL_DEPTH_FORWARD_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n6a_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n6a.json` | protected check |
