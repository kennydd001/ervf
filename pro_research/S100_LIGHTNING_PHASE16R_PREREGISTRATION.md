# S100 Lightning Phase 16R preregistration
Source evidence: `agent/s100-lightning-phase16-hardware` at commit `3c0418f`.


## Frozen diagnosis

Before this run, recovery is defined by candidate identity, not filename:

`(terms, sorted(cases), handoff)`

An orchestration bug is confirmed if a Phase-16 calibration file matching that
identity is measured and strict-green while the Phase-16 selection file has no
quality object for the canonical screen candidate.

Recovered evidence only restores candidate eligibility. Every candidate is
rerun through a clean, stably named Phase-16R calibration before validation can
open.

## Quality gates

Use the existing Phase-16 `STRICT` calibration/validation gates and `OFFICIAL`
heldout gates unchanged. Validation and heldout traces remain untouched until
their predecessor gate passes.

## Calibration-only K/V repair

Terms are fixed at TC1. O remains excluded.

1. Evaluate all 12 K/V matrices together with one producer synchronization per
   attention layer when K and V are paired.
2. Evaluate every leave-one-out set.
3. If none passes strict calibration, rank those trials by frozen normalized
   gate pressure, take the six best removed matrices, and evaluate all
   leave-two-out combinations among those six.
4. Select at most two strict-green novel candidates, maximizing native weight
   bytes first and minimizing gate pressure second.

No validation information participates in subset selection.

## Throughput

Use all ten calibration prompts and their frozen Phase-15 Lightning target
tokens. After eight warmup target steps per prompt, time the remaining 56
teacher-forced steps, yielding 560 samples per arm.

Arms:

- current production CUDA graph parent;
- current eager parent;
- each heldout-green candidate using its measured handoff.

A net-speed gate opens only when both aggregate tokens/s and median latency are
at least 3% better than the graph parent. `S100_SINGLE_ACHIEVED` additionally
requires >=100 aggregate tok/s with heldout quality green.

## Claim boundary

Phase 16R can recover the selective-native route. It cannot reopen the measured
~54 tok/s perfect-draft verifier or the failed Lightning DFlash2 transfer and
memory gates.
