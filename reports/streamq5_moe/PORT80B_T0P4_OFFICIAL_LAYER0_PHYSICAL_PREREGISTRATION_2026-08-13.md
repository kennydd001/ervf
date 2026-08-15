# PORT80B-T0-P4 — immutable official layer-0 real-Q5 physical preregistration

Date: 2026-08-13  
Phase: physical candidate, closed until T0-R4 passes independently  
Target revision: `a19358a7659bd1f564300250ee189120c49a562f`

This is the immutable independent-audit repair after T0-P3. Older documents
remain evidence. P4 keeps its model, physical split, timing and claim boundary,
but separates source/numerical evidence from mapped-transport exactness and
strengthens lifecycle, cold-route, telemetry and eligibility contracts.

## Falsifiable question and exact claim

Can the proven 499-mapped + 13-pageable mechanism execute official, real,
differentiated layer-0 Q5 expert weights on the held-out natural top-10 routes
from T0-R4 and reproduce its independent real-Q5 evidence under the separately
frozen source, numerical and mapped-transport oracles?

A pass proves a target-valid official **layer-0 expert-plane** result. Unless a
future runner explicitly executes it, this does not claim the Gated-DeltaNet
itself runs on the GPU. It does not prove full-depth logits, quality,
tokens/second, endurance, deployment readiness or an industrial breakthrough.

## Eligibility and immutable inputs

T0-P4 is closed unless every T0-R4 gate passes and a separate CPU verifier
rehashes all raw bytes. Freeze the T0-R4 result, prompt lock, environment lock,
official index/shard, quantizer, bank builder, physical runner, verifier and
this preregistration by SHA-256. The current T0-R4 digest is bound by the R4/P4
preflight rather than copying an obsolete preregistration digest as current.

The D9/D10 499+13 registration lifecycle, stream dependencies, width-8 Q5
kernel structure, telemetry and integrity scaffolding may be adapted. Their
synthetic weights, routes, recurrent shell and outputs are forbidden as target
numerical evidence.

## Exact real bank

Build one new immutable layer-0 bank from official shard 1 after T0-R4:

- 512 routed experts followed by one shared expert;
- three 675,840-byte projection records per expert;
- 2,027,520 bytes per expert record;
- total: **1,040,117,760 bytes**;
- registered routed prefix 0–498: **1,011,732,480 bytes**;
- pageable routed tail 499–511: **26,357,760 bytes**;
- shared expert: **2,027,520 bytes**, separate resident/mapped allocation and
  never counted in prefix capacity.

Build to `.inprogress`, fsync, rename atomically, then emit a manifest with
full bank/source/record/header/codes/scales/payload SHA-256 and CRCs. Require at
least 95% unique routed expert payload triples. Map the final bank read-only.
Do not rebuild or scan the deleted/old 49.9-GB synthetic bank.

Expected added disk is about 1.05 GB plus raw artifacts; require 20 GiB free.
Require 8 GiB available RAM before registration, 2 GiB reserve throughout,
and 512 MiB free VRAM after allocation. Selected ten expert records occupy
20,275,200 bytes; the shared record occupies 2,027,520 bytes. A full-bank HBM
copy, replication, mutation or hidden weight cache is forbidden.

## Frozen cases and timing

- Debug only: four prompts, positions 0–7.
- Primary: the exact 32 T0-R4 held-out rows at positions 8–15 in locked order.
- Mechanism controls: one all-hot row and one forced row containing all cold
  experts 499–511, excluded from natural-route quality.
- Exactly eight warm-ups over calibration rows.
- Exactly ten repetitions per primary row: 320 inclusive wall measurements.

Inclusive wall time begins before pointer construction and any pageable cold
copy and ends after the composed output is host-visible. Report per-token and
aggregate p50/p95/p99; CUDA event timing is diagnostic only. The exploratory
component gate is p95 at most 15 ms and maximum at most 30 ms. A timing failure
cannot erase an exactness pass; it yields the explicit performance-negative
outcome below.

## Raw evidence and controls

For every held-out row retain raw candidate gate, up, SwiGLU, per-expert down,
weighted routed sum, shared raw/gate/gated and composed output arrays. Retain
route IDs/weights, source expert IDs, record pointers/slot IDs, headers and
payload hashes. Source-to-record decode identity remains independently exact.

Mapped/pageable transport exactness compares raw arrays bitwise to a
resident-device execution using the same frozen CUDA kernel, launch geometry,
inputs, reduction order and cast points. CPU numerical comparison is separate:
for each FP32 dot of length `n`, use
`gamma(2n)=(2n*2^-24)/(1-2n*2^-24)` times `sum(abs(x_i*w_i))`, add propagated
input error and `0.5*ULP_BF16` at each BF16 cast; use `n=2048` for gate/up and
`n=512` for down. Propagate intervals through SiLU using a predeclared
derivative bound and add one BF16 ULP for backend transcendental error. This
formula is frozen before values, while observed magnitudes may populate it.
Native Torch CPU is diagnostic only. Any CUDA-order change needs a new prereg.

Before physical use, independently decode every five-bit field as
`q = field - 15`; reject field 31. The source contract must prove that builder,
CPU oracle and CUDA decoder all implement this mapping and never interpret the
payload as two's complement.

Wrong-layer substitution is impossible with this layer-0-only bank and is not
claimed. On at least eight held-out rows run all four controls:

1. substitute rank-0 with a different expert while retaining its weight;
2. substitute expert 498 for 499 or vice versa (or the deterministic nearest
   present hot/cold IDs if neither is natural);
3. permute two pointer-table slots but retain route IDs and weights;
4. present a valid record whose header expert conflicts with manifest identity.

Every control must be rejected by identity/integrity logic and change the raw
composed-output digest with at least one different BF16 word.

## Conjunctive hard gates

1. T0-R4 and independent raw-byte verification pass; every lock is current,
   including whole-prefix/token-step state equivalence, raw cache-state schema,
   and manual-MoE/official-layer maximum-one-BF16-ULP equivalence.
2. Bank is exactly 1,040,117,760 bytes; all 513 records rehash/CRC correctly;
   identity and uniqueness gates pass.
3. Natural IDs and normalized weights equal T0-R4 exactly; selected source
   identities bind layer 0 and the intended expert/shared record.
4. All mandatory arrays are fully written and finite. Resident-device versus
   mapped/pageable arrays have zero bit differences on all 32 held-out rows;
   CPU source/decode checks are exact and CPU numerical intervals all contain
   the CUDA outputs.
5. All four adversarial controls are positively detected on every control row.
6. Both mapped 499 prefix and pageable 13 tail are exercised. Report whether
   any natural held-out route selects IDs 499–511. If none does, cold evidence
   is mechanism-control only and may not be described as natural. No hidden copy.
7. Exactly 320 finite positive inclusive timings; report pass or explicit
   performance negative using the frozen 15/30-ms gates.
8. RAM never below 2 GiB, free VRAM never below 512 MiB after allocation.
   Require at least 30 post-warm-up 1-Hz page-telemetry samples to apply the
   p95 2,048/no-sample-above-8,192 gates; otherwise page telemetry is diagnostic.
   Repeated timings are warm-page timings unless a separate legitimate cache
   state protocol proves otherwise. No
   CUDA, driver, runner or integrity error.
9. Exactly one successful prefix registration and its successful matching
   unregister; record every attempt, return code and error. Register followed
   by unregister failure is a lifecycle failure. All
   all buffers/handles close and at least 2 GiB RAM remains after cleanup.

## Frozen outcomes

- `real_layer0_exact_component_pass`
- `real_layer0_exact_performance_negative`
- `real_layer0_correctness_negative`
- `blocked`
- `invalid`

No T0-P4 outcome automatically opens endurance or a full-depth claim. The next
gate is separately preregistered official full-depth execution with final norm,
LM head, vocabulary logits, held-out cross-entropy/top-1 agreement, and natural
routes at every layer.
