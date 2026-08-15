# D10A1 first-touch independent source audit

**Verdict:** `50_GiB_start_gate_is_overconservative_for_the_frozen_component_trace`  
**Method:** CPU-only source inspection and exact replay of frozen P4D-shaped route lifting  
**GPU/bank action:** none

## Exact unique mmap footprint

The source selects 40 correctness cases (`5 domains x tokens 0..7`) and the
first 32 validation cases (`general`, tokens 512..543). Each case has 48 layers
and top-10 records. Replaying the runner's exact `lift()` function over the
hash-locked route tensors gives:

| source path | unique records | 4-KiB pages | bytes | GiB |
|---|---:|---:|---:|---:|
| 40 correctness cases | 11,329 | 5,607,855 | 22,969,774,080 | 21.392269 |
| 32 measured validation cases | 6,953 | 3,441,735 | 14,097,346,560 | 13.129177 |
| correctness + validation union | 14,404 | 7,129,980 | 29,204,398,080 | 27.198715 |
| 48 shared records | 48 | 23,760 | 97,320,960 | 0.090637 |
| layer-0 reference records 0..9 | 10 | 4,950 | 20,275,200 | 0.018883 |
| two negative override sources | 2 | 990 | 4,055,040 | 0.003777 |
| **complete unique union** | **14,452** | **7,153,740** | **29,301,719,040** | **27.289352** |

The bottom line is a 27.289-GiB worst-case unique file-page footprint for the
frozen component trace. The 10 layer-0 reference records and both negative
override sources already occur in the correctness/validation union, so they do
not add unique pages. The correctness and validation partitions overlap by
3,878 `(layer, expert)` records.

The eight validation warm-ups do not enlarge the union because they reuse the
first eight of the same 32 validation cases. Repeated reads can affect timing
and residency order, but not unique first-touch bytes.

For completeness, the runner issues 59,098 whole-record mmap read occurrences
before/through validation:

- 38,400 correctness reads: each case staged once and assembled independently
  into the oracle once;
- 480 reads for the first-route resident oracle used by negative controls;
- 960 reads for two wrong-pointer stages;
- 48 shared-record reads and 10 layer-0 reference-record reads;
- 3,840 reads in eight warm-ups;
- 15,360 reads in 32 measured validation steps.

These repetitions total about 111.6 GiB of logical record traffic, but only the
27.289-GiB union can first-touch distinct bank pages.

## Hidden full-bank scan audit

No hidden full-bank scan exists in the D10A1 component source:

- `audit()` opens a read-only `np.memmap` only to confirm mode and exact size;
  it does not index or hash the bank contents;
- the compile evidence explicitly records `bank_scan: false`;
- `component_phase()` copies only route-selected records, 48 shared records and
  ten layer-0 reference records;
- `sha256()` is used for small provenance files, route tensors and manifests,
  never for `BANK`;
- route-tensor hashing/loads total only a few MiB and do not touch bank pages.

Registering 48 x 499 prefixes covers 45.228 GiB of virtual file ranges, but it
is not itself a deterministic first-touch of those pages. D9 measured almost no
available-RAM loss at registration and the large drop only after execution.
The safety calculation should therefore use the exact routed union while still
retaining post-registration and post-warm-up hard stops.

## Safe start-RAM gate

The deterministic lower bound that leaves a 2-GiB final reserve is:

```
29,301,719,040 unique first-touch bytes
+2,147,483,648 required reserve
=31,449,202,688 bytes = 29.289352 GiB
```

To cover ordinary OS/background variability without confusing it with bank
first touch, use a 1-GiB explicit safety margin:

```
recommended start gate = 32,522,944,512 bytes = 30.289352 GiB
```

The observed resource-stop state, 50,252,566,528 bytes (46.801 GiB), exceeds
this recommended gate by about 16.512 GiB. The locked 50-GiB gate exceeds the
recommended source-derived gate by about 19.711 GiB and is therefore
overconservative for this exact component trace.

This recommendation is safe only with all existing dynamic interlocks retained:

1. at least 2 GiB still available immediately after registration;
2. run the eight frozen warm-ups, then require at least 2 GiB after first touch;
3. abort below 1.5 GiB at every measured validation checkpoint;
4. fail if validation loses more than 1 GiB from first to last telemetry point;
5. no endurance authorization from this audit; endurance has a different route
   union and must receive its own first-touch calculation.

An even safer implementation would precompute and store the exact 14,452-record
union digest before GPU work, then use its byte count in the gate. That is a
small metadata operation over route IDs, not a bank scan.

## Cross-check against adjacent audit evidence

The result agrees with the adjacent architecture/new-pack limits:

- P4D remains explicitly labelled `p4d_shaped_synthetic_proxy`, not a natural
  512-expert target trace;
- D9 already showed registration does not necessarily fault all mapped pages;
- the component preregistration separately requires 2 GiB after first touch and
  an emergency 1.5-GiB floor;
- no conclusion here changes model, quality, endurance or real-checkpoint claim
  boundaries.

This is a source-derived safety audit, not experimental evidence that the
component will pass timing, paging or correctness.
