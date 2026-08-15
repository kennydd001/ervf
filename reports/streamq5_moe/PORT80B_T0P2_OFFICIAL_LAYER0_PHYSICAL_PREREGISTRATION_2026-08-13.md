# PORT80B-T0-P2 — immutable official layer-0 real-Q5 physical preregistration

Date: 2026-08-13  
Phase: physical candidate, closed until T0-R2 passes independently  
Target revision: `a19358a7659bd1f564300250ee189120c49a562f`

This is the immutable codec correction after T0-P and T0-P1. Older documents
remain evidence. The sole semantic repair is the exact proven STREAMQ5 wire
mapping: quantized `q in [-15,15]` is stored as `q+15 in [0,30]`, in little
order as eight five-bit fields per five bytes; field 31 is invalid. Provenance,
numerical thresholds, physical gates, resources and timing are unchanged.

## Falsifiable question and exact claim

Can the proven 499-mapped + 13-pageable mechanism execute official, real,
differentiated layer-0 Q5 expert weights on the held-out natural top-10 routes
from T0-R2 and reproduce its independent dequantized-Q5 oracle bit-for-bit?

A pass proves a target-valid official **layer-0 expert-plane** result. Unless a
future runner explicitly executes it, this does not claim the Gated-DeltaNet
itself runs on the GPU. It does not prove full-depth logits, quality,
tokens/second, endurance, deployment readiness or an industrial breakthrough.

## Eligibility and immutable inputs

T0-P2 is closed unless every T0-R2 gate passes and a separate CPU verifier
rehashes all raw bytes. Freeze the T0-R2 result, prompt lock, environment lock,
official index/shard, quantizer, bank builder, physical runner, verifier and
this preregistration by SHA-256. The current T0-R2 digest is bound by the R2/P2
preflight rather than copying the obsolete T0-R1 digest as current.

The D9/D10 499+13 registration lifecycle, stream dependencies, width-8 Q5
kernel structure, telemetry and integrity scaffolding may be adapted. Their
synthetic weights, routes, recurrent shell and outputs are forbidden as target
numerical evidence.

## Exact real bank

Build one new immutable layer-0 bank from official shard 1 after T0-R2:

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
- Primary: the exact 32 T0-R2 held-out rows at positions 8–15 in locked order.
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
payload hashes. Compare every stored array with T0-R2's Q5 oracle under the
frozen rounding rules; primary correctness requires zero differing bits.

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

1. T0-R2 and independent raw-byte verification pass; every lock is current.
2. Bank is exactly 1,040,117,760 bytes; all 513 records rehash/CRC correctly;
   identity and uniqueness gates pass.
3. Natural IDs and normalized weights equal T0-R2 exactly; selected source
   identities bind layer 0 and the intended expert/shared record.
4. All mandatory arrays are fully written, finite and bitwise equal to the
   independent Q5 oracle across all 32 held-out rows: zero bit differences.
5. All four adversarial controls are positively detected on every control row.
6. Both the mapped 499 prefix and pageable 13 tail are exercised. Every natural
   cold selection reads the intended pageable record. No hidden full copy.
7. Exactly 320 finite positive inclusive timings; report pass or explicit
   performance negative using the frozen 15/30-ms gates.
8. RAM never below 2 GiB, free VRAM never below 512 MiB after allocation,
   post-warm-up Page Reads/s p95 at most 2,048 and no sample above 8,192; no
   CUDA, driver, runner or integrity error.
9. Exactly one successful prefix registration and one successful unregister;
   all buffers/handles close and at least 2 GiB RAM remains after cleanup.

## Frozen outcomes

- `real_layer0_exact_component_pass`
- `real_layer0_exact_performance_negative`
- `real_layer0_correctness_negative`
- `blocked`
- `invalid`

No T0-P2 outcome automatically opens endurance or a full-depth claim. The next
gate is separately preregistered official full-depth execution with final norm,
LM head, vocabulary logits, held-out cross-entropy/top-1 agreement, and natural
routes at every layer.
