# PORT80B-T0-P — official layer-0 real-Q5 499+13 physical preregistration

Date: 2026-08-13  
Phase: physical candidate, closed until T0-R independently passes

## Question and claim

T0-P asks whether the already demonstrated 499-mapped + 13-pageable mechanism
correctly executes **real, differentiated Q5 weights** for the natural layer-0
top-10 routes captured by T0-R and reproduces the independently stored CPU Q5
oracle.

A pass is a target-valid official layer-0 physical expert-plane result. It is
not a full layer-0 Gated-DeltaNet GPU result unless the candidate explicitly
executes and compares that graph, and it is never a full-model quality,
tokens-per-second, endurance or breakthrough claim.

## Eligibility and immutable inputs

T0-P is closed unless T0-R passes every gate and an independent CPU verifier
rehashes every retained raw tensor. Freeze T0-R result/reference-artifact,
prompt-lock, reference-environment, official shard/index, quantizer, bank
builder, candidate runner and this preregistration by SHA-256.

The existing D9/D10 transport and width-8 Q5 kernel structure, 2,027,520-byte
expert-record layout, stream/lifecycle discipline, PDH/process telemetry and
499+13 split may be reused. The 49.9-GB uniform synthetic bank and its payload
may not be used for numerical evidence.

## Bank build and resource scope

Build exactly one immutable layer-0 bank from T0-R's real official tensors:

- 512 routed + one shared expert = 513 records;
- 3 matrices per expert, each 675,840 bytes;
- expert record = 2,027,520 bytes;
- total bank = **1,040,117,760 bytes**;
- mapped registered prefix experts 0–498 = **1,011,732,480 bytes**;
- pageable cold experts 499–511 = **26,357,760 bytes**;
- shared expert = **2,027,520 bytes**, resident device or separately mapped as
  preregistered, but never counted as routed prefix capacity.

Build to `.inprogress`, fsync, rename atomically, then write a manifest with
full bank SHA-256, source shard/index/revision, quantization semantics,
per-record header/source/codes/scales/payload hashes and CRCs. Map read-only.
No 49.9-GB rebuild and no full checkpoint are required.

Expected additional disk is approximately 1.05 GB for bank plus retained raw
T0-R/T0-P tensors; require ≥20 GiB free before build. Require ≥8 GiB available
RAM before registration, ≥2 GiB throughout, and ≥512 MiB free VRAM after all
candidate allocations. Register exactly one 499-record prefix; record every
registration/unregistration attempt and cleanly unregister exactly once.

## Frozen cases

- Calibration/debug: T0-R prompt positions 0–7; outputs may diagnose only.
- Primary held-out: the 32 T0-R positions 8–15, in frozen prompt/token order.
- Mechanism controls: one all-hot row and one forced row containing every cold
  expert 499–511, reported separately and excluded from natural-route quality.
- Warm-ups: exactly eight iterations over calibration rows.
- Measurements: exactly 10 repetitions per held-out row (320 inclusive wall
  rows). Report per-token and aggregate p50/p95/p99; do not mix warm-ups.

Inclusive wall time begins before pointer-table construction/cold copies and
ends after composed output is host-visible. CUDA events are diagnostic only.

## Correctness and adversarial controls

For every held-out row retain raw candidate arrays for gate, up, SwiGLU,
per-expert down, weighted routed sum, shared raw/gate/gated output and composed
output. Compare every array with T0-R's dequantized-Q5 oracle using exact frozen
rounding; primary correctness requires zero differing bits. Retain source
expert IDs, record pointers/slot IDs, header fields and record payload hashes.

Wrong-layer control is impossible with a shard-1/layer-0-only bank and is not
claimed. Instead run all of these on at least eight held-out rows:

1. wrong-expert substitution at rank 0, preserving the original weight;
2. hot/cold-boundary substitution (expert 498 ↔ 499, or deterministic nearest
   available hot/cold IDs if neither is naturally selected);
3. slot permutation: swap two pointer-table slots while retaining original
   route IDs and weights;
4. header/source mismatch: present an otherwise valid record whose header ID
   disagrees with the source-tensor identity locked in the manifest.

Every control must be detected by both identity/integrity checks and a changed
composed-output digest with at least one differing BF16 word. A control that
does not affect output closes the gate; it may not be dropped post hoc.

## Hard gates

1. T0-R and its independent raw-byte verifier pass; all locks current.
2. Exact 1,040,117,760-byte differentiated bank; all 513 records independently
   rehashed and CRC-valid; ≥95% unique expert payload triples.
3. Natural route IDs and normalized weights exactly equal T0-R; all selected
   record source identities match layer 0 and intended expert.
4. All stored candidate intermediate and composed arrays are fully written,
   finite and bitwise equal to the T0-R dequantized-Q5 oracle: zero differing
   bits over all 32 held-out positions.
5. All four adversarial controls above are positively detected on every
   preregistered control row.
6. The 499 mapped prefix and 13-pageable tail are both exercised. All natural
   cold selections source the correct pageable record; no hidden full-bank
   HBM copy, weight replication, or mutation is permitted.
7. Exactly 320 finite positive inclusive wall measurements. Exploratory
   timing gate: p95 ≤ 15 ms for the layer-0 expert plane and maximum ≤ 30 ms.
   This threshold is a component diagnostic and may not be extrapolated to 48
   layers or tokens/s. Correctness can pass even if timing fails; the outcome
   must then be `real_layer0_exact_performance_negative`.
8. Available RAM never <2 GiB; post-allocation free VRAM never <512 MiB;
   post-warm-up Page Reads/s p95 ≤2,048 and no sample >8,192; no CUDA, driver,
   runner or integrity error.
9. Exactly one successful prefix registration and one successful unregister;
   shared/cold allocations freed, file handles closed, and available RAM ≥2
   GiB after cleanup. Cleanup failure invalidates the result.

## Outcome vocabulary

- `real_layer0_exact_component_pass`: correctness, controls, physical path and
  resources pass; timing gate may be separately labeled pass/negative.
- `real_layer0_exact_performance_negative`: exactness passes, timing fails.
- `real_layer0_correctness_negative`: any numerical/identity/control failure.
- `blocked`: missing/incompatible provenance, reference, resource or toolchain;
  no mechanism conclusion.
- `invalid`: protocol, overwrite, hidden-copy, cleanup or split violation.

No T0-P outcome opens full-depth quality or endurance automatically. The next
eligible gate after a pass is a separately preregistered multi-layer/full-depth
official checkpoint run with final norm/LM head, reference vocabulary logits,
held-out cross-entropy/top-1 agreement and natural routes at every layer.

