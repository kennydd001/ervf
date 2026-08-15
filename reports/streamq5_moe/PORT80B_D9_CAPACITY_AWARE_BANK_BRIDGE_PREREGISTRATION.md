# PORT80B-D9 — capacity-aware 499+13 bank bridge preregistration

**Frozen before compile/preflight and physical execution:** 2026-08-12

## Question

D8 established only a 499-routed-expert prefix per layer as cleanly
registerable: the 512 row ended with 44/48 unregister failures and the sweep
had cumulative-memory-state bias. D9 asks whether the exact D7 staged Q5 plane
can bridge that measured capacity boundary without pretending that the last 13
routed experts are registered.

## Immutable inputs and non-overwrite rule

- Existing 49,925,652,480-byte PORT80B-P0 synthetic Q5 bank and its manifest.
- Expected bank SHA-256 from the immutable manifest:
  `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`.
- Existing D7 `host_to_smem_pipeline`, width-8 exact Q5 gate/up/down kernels,
  and canonical BF16 SwiGLU.
- Existing D8 independent conclusion: 499/layer is the largest clean prefix;
  512 is not a clean pass.
- Every D9 output path is new and the evaluator must refuse overwrite. The
  bank is mapped read-only. No bank bytes or central registry are modified.

## Bridge mechanism

Exactly 499 routed records per layer (experts 0…498) are registered as 48
read-only mapped ranges. Hot source pointers are derived only from those 48
mapped aliases. Experts 499…511 are the explicit cold tail and may never be
addressed through a registered alias. Every cold occurrence is copied from its
pageable read-only bank address into a dedicated HBM escape slot before the
D7 staging kernel. The staging table then mixes mapped-hot pointers with HBM
cold-escape pointers and stages exactly 480 complete expert records into the
existing 973,209,600-byte D7 work buffer. Copy, staging, and exact Q5 compute
are serial on one nonblocking stream; no overlap or cache credit is claimed.

## Differentiated route-integrity oracle

The payload pattern is invariant, so payload-only comparison cannot detect a
wrong layer or expert. D9 therefore verifies every byte of the 480-record
staged image against the bank’s differentiated 64-byte `SQ5M` headers plus the
known code, scale, and padding contract. Headers encode layer, expert, and
projection. Required controls before timing are:

1. zero full-image byte mismatches for each positive route case;
2. a deliberately substituted same-layer/wrong-expert pointer must produce at
   least one mismatch;
3. a deliberately substituted wrong-layer/same-expert pointer must produce at
   least one mismatch;
4. all three exact Q5 outputs must be bitwise identical to the resident D7
   oracle, with equal SHA-256 and finite values.

Any failure closes timing and test.

## Frozen route cases

Every case contains ten unique routed experts at each of 48 layers:

- `all_hot`: ten deterministic experts from 0…498 using the frozen PORT80B
  route generator and token 130001; 480 hot, 0 cold occurrences.
- `mixed_5_hot_5_cold`: the first five deterministic prefix experts from token
  130002 and five tail experts `499 + ((layer + rank) mod 13)` for ranks 0…4;
  240 hot, 240 cold occurrences.
- `all_cold_tail`: tail experts `499 + ((layer + rank) mod 13)` for ranks 0…9;
  0 hot, 480 cold occurrences.

No route selection or schedule is tuned after observing results.

## Execution protocol and hard safety stops

1. `--phase compile`: read-only artefact/stat audit, Python compilation, CUDA
   source compilation and symbol resolution only. No host registration, large
   HBM buffer, timing, or bank sweep.
2. `--phase run`: refuse if compile evidence is missing/negative, any result
   path exists, bank/manifest contract fails, or available physical RAM is
   below 2 GiB before registration.
3. Register exactly 48 × 499 prefixes. Refuse timing if fewer than 48 mapped
   aliases exist or available physical RAM falls below 2 GiB afterward.
4. Run route-integrity controls and bitexact captures.
5. Four untimed warm-ups per case, then 24 validation samples per case in a
   fixed rotating/reversing case order. Timings are inclusive wall-clock
   milliseconds from before cold escape submission through final stream
   synchronization; CUDA-event timings are recorded diagnostically.
6. Test opens only if all correctness/integrity controls pass, every validation
   sample is finite, and validation p50 is at most 65/100/135 ms for
   all-hot/mixed/all-cold respectively.
7. If opened, run 60 once-only test samples per case in the same fixed
   rotating/reversing order. No retuning.
8. Synchronize and unregister all 48 ranges in reverse order. Any unregister
   failure makes the final result negative regardless of earlier measurements.

## Validation and test gates

Primary pass requires all of:

- bank mapping is read-only; prefix is exactly 499/layer; cold tail is exactly
  13/layer; source-provenance counts match each frozen case;
- all three positive header/payload checks have zero mismatches and both
  differentiated negative controls are detected;
- all three output captures are finite, bitexact, and SHA-equal to the resident
  oracle;
- validation opens test under the frozen p50 limits above;
- 60 finite test wall timings per case;
- test wall p95 ≤65 ms all-hot, ≤100 ms mixed, and ≤135 ms all-cold-tail;
- 48 successful registrations, 48 clean unregisters, no CUDA/runner error, and
  ≥2 GiB available RAM immediately after registration.

Strong pass additionally requires mixed test p95 ≤80 ms and all-cold-tail test
p95 ≤110 ms. Failure of validation leaves the test arrays empty. A cleanup
failure overrides all pass labels.

## Claim boundaries

D9 can establish only an exact, physically timed synthetic 499+13 capacity
bridge for one frozen width-10 active plane. It does not repair D8’s 512-row
failure and does not establish full-bank pinning, long-duration registration,
concurrent cold copies, a real checkpoint, natural routing frequency, quality,
a physical dense shell, end-to-end tokens/s, energy, or endurance. The
all-cold case is an adversarial mechanism bound, not a traffic-distribution
claim. CUDA-event timing is diagnostic; pass/fail uses inclusive wall time.
