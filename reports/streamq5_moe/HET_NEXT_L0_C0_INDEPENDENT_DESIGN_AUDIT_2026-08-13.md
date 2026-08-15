# HET-NEXT-L0-C0 independent design audit

## Verdict

**NO-GO for implementation or any capability/device phase.** The component
hypothesis, whole-expert split and claim boundary are reasonable, but the
frozen protocol has four specification blockers that can change correctness or
the performance verdict. A revised preregistration must close them before a
runner, compiler or executable preflight is frozen.

No device was enumerated or opened and no kernel, model, checkpoint forward or
timing arm was run for this audit.

## Checks that pass

- Both audited documents match their supplied hashes:
  `5ba80f6c8f3a5b144192146dde32a1e3b8e0439a60e0370afa049623a6e8cd63`
  and `86a8f60eb7779ff9951c4bf7611406518d37a86626fb70ecf4e4cff36b4e7495`.
- Every listed D2-R3, R5, C1-R2A, shard, ST2-mini and D7 hash exists locally
  and matches. Shard 1 is exactly `3,999,619,288` bytes with the declared SHA.
- The four ten-expert route lists exactly match D2-R3 row index 15; route
  weights and shared-gate captures are BF16.
- Intel ranks 0--3 and NVIDIA ranks 4--9 plus shared is a fixed whole-expert
  split. The three arms use the same logical split/merge boundary and do not
  inherit ST2 or D7 as a performance pass.
- The claim boundary correctly excludes official-layer/logit equality,
  held-out quality, end-to-end throughput, full-model acceleration and a
  breakthrough claim. It preserves R5 as formally negative and C1-R2A as a
  synthetic sensitivity result only.
- The phased no-device -> capability-only -> CPU source-build -> separately
  locked validation structure is directionally sound. Intel host-USM is
  locally evidenced by ST2-mini, but that evidence does not replace a fresh
  capability probe or a real-weight full-SwiGLU oracle.

## Blocking repairs

### 1. The routed BF16 accumulation order is not the official implementation order

The preregistration mandates accumulation in route-rank order `0..9`. The
bound Transformers implementation constructs `expert_hit` and iterates expert
IDs in ascending order, then uses the original route position only to select
the BF16 weight before BF16 `index_add_`. The R5 runner/verifier reproduces the
same expert-ID order. For p0 the official order is therefore ranks
`7,9,0,8,1,2,6,4,5,3`, not `0..9`.

Repair: preregister two distinct immutable notions: dispatch ownership remains
route-rank based, but host accumulation must be ascending expert ID with the
captured weight from that expert's original rank. Freeze every BF16 cast and
rounding point: Q5 GEMV output, SiLU/up product, down output, route-weight
multiplication, each sequential BF16 add, `sigmoid(shared_gate) * shared_raw`,
and final routed-plus-shared add. The independent CPU oracle and both device
arms must use this exact order.

### 2. The validation/test sealing rules contradict the source-build phase

The preregistration says p1--p3 remain unopened until p0 passes, and Phase 0
requires a test-data read prohibition. Phase 2 nevertheless rereads all four
test inputs/routes/shared gates, builds the four-row union and permits all four
CPU oracle arrays before validation. That makes the state machine ambiguous,
even though no held-out claim is made.

Repair: define precisely what is public before validation. A safe bounded
choice is to allow only the already preregistered route-ID lists for union
weight construction, while forbidding p1--p3 input, route-weight,
shared-gate, oracle, control and timing payload reads until p0 commits a pass.
Alternatively remove the unopened/test language and honestly call all four
rows fixed validation replicates. The current mixture is invalid.

### 3. Sample count, pairing and ratio arithmetic are underdetermined

`120 paired timed samples per row` conflicts with a three-arm schedule whose
12-observation blocks have equal arm counts. It is not stated whether 120 is
per arm or total, how three arms are paired, whether the gate is a ratio of
arm percentiles or a percentile of paired ratios, which quantile convention is
used, or what exact Williams/ABBA seed and sequence apply.

Repair: freeze the complete schedule and digest before capability execution;
state total observations and observations per arm; define pair IDs; define the
quantile algorithm; and write the exact four gate formulas. A suggested
unambiguous design is 120 observations per arm (360 per row), a frozen
three-treatment balanced Williams schedule, and separately computed arm p50
and p95 followed by `Q_hybrid / Q_dGPU` at each quantile. No pooled or pairwise
alternative may be selected later.

### 4. Cache, paging and clock-collapse gates are not executable as written

The cache-thrash size/access pattern and record rotation are not frozen;
`paging` has no counter or threshold; and the real-time 70%-for-five-samples
clock stop refers to the validation median, which does not yet exist while the
validation sequence is running.

Repair: freeze exact thrash-buffer bytes, NUMA allocation, read/write stride,
per-sample order and digest; name the Windows/system and device paging counters
and zero/nonzero adjudication; and define a pre-timing clock baseline (for
example the ten frozen warmups) usable during validation. Store raw counters
for every sample and prohibit changing these rules after p0.

## Required next gate

Publish a C0-R1 preregistration and Phase-0 design containing only these
repairs, then repeat an independent no-device source audit. Until that audit is
GO, no executable preflight, device enumeration, capability kernel or physical
attempt is authorized.
