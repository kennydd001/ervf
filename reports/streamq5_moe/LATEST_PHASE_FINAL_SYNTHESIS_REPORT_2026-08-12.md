# Latest-phase synthesis — STREAMQ5-MoE

**State:** final through D10B-R, twice independently verified  
**Evidence finalized:** 2026-08-13  
**Registry:** `LATEST_PHASE_COMBINED_EVIDENCE_REGISTRY_2026-08-12.json`  
**Central historical registry changed:** no

## Bottom line

The strongest result is now D10B-R: a synthetic, held-out 10,000-step endurance
pass of the 499-mapped + 13-pageable DirectPath bridge inside a physically
allocated, shape-informed component shell. It passes all 19 frozen gates and
two independent CPU audits. Inclusive wall p50/p95/p99 is
67.011950/69.047360/77.218634 ms; maximum is 96.345300 ms. The last-1,000 p95
is 69.031910 ms versus 79.345895 ms for the first 1,000, a ratio of 0.870012.
All 10,000 composed-state guards and telemetry rows pass. All 48 mapped ranges
register and unregister cleanly.

This is meaningful systems evidence: D9's one-step capacity bridge survives
first touch, physical coexistence with allocated dense/KV/recurrent/runtime
state, held-out route variation and a sustained 719.281-second run. It is not
an LLM or industrial breakthrough. The bank uses uniform synthetic Q5 payloads,
the routes are P4D-shaped proxies, the shell is shape-informed rather than an
official model implementation, and no retained language-model quality or real
checkpoint decode rate was measured.

## Evidence progression

| Phase | Immutable outcome | Decisive evidence | Boundary |
|---|---|---|---|
| D1–D4R | verified negative/invalid attempts preserved | token batching, ordinary scatter, naive mapped-host copy and batch copy fail their frozen gates | does not test staged Q5 |
| D5 | strong transport component pass | 973.21 MB mapped-host → shared-memory → HBM at 43.708-ms p95 and 22.266 GB/s | no Q5/model/endurance |
| D6 | exact, performance negative | direct host Q5 is bitexact but validation p95 is 77.074 ms | motivates staging |
| D7 | strong staged-Q5 component pass | 0 bit differences, p95 49.977 ms, projected shell p95 78.055 ms | uniform payload cannot catch wrong expert numerically |
| D8 | capacity diagnostic | 499/512 experts per layer is the largest clean observed prefix; the 512 arm is invalid due to unregister OOM failures | not a stable full-registration guarantee |
| D9 | independently verified capacity-bridge pass | truthful 499 mapped + 13 cold escape covers all expert indices; hot/mixed/cold p95 49.116/68.670/88.136 ms | one-step synthetic test only |
| D10A1-R | immutable negative | 18/21 gates pass; 491 header-verifier mismatches, zero conv words and 96,256/98,304 shared-output differences | later audit finds a common cross-stream race, but prior run remains negative |
| D10A2-R | immutable blocked/negative execution | 48/48 register/unregister cleanup succeeds, then a 48-vs-36-layer CPU oracle assertion stops before validation | experimental-plumbing failure, no mechanism conclusion |
| D10A2-R2 | independently verified component pass | 28/28 gates; 40 exact Q5 cases; exact 292,608-word conv oracle; 98,304 shared outputs bitexact; p95 91.205 ms | component evidence only; endurance separately gated |
| original D10B | immutable CPU-preflight failure | inherited unit provenance compared against the wrong runner/prereg; every physical action false | no GPU execution or result |
| D10B-R | twice independently verified endurance pass | exact 10,000 held-out steps; 19/19 gates; 10,000 state checks; 101 checkpoints; clean 48+48 lifecycle | summary evidence, synthetic payload/routes/shell |

Other routes retain their earlier outcomes: co-route physical ordering is a
verified validation negative; SplitTree ST2-mini executes exact Intel host-USM
Q5 but misses the 21.63-GB/s tail gate; TierFlow's fixed eight-expert functional
span fails its frozen functional gates; Nemotron remains metadata/header-only
with no payload execution or proven NIM alias identity.

## What D10A2-R2 proves

The deterministic single-stream repair removes the common race in uploads,
clears, verifier counters and component outputs. The repaired component run has:

- 40/40 routed-Q5 candidate/oracle comparisons bitexact with zero header or
  canary mismatches and 40 unique output digests;
- wrong-expert and wrong-layer controls each detected both in headers and in
  2,050 numerical output bits;
- a complete step-zero GDN conv oracle with 292,608 nonzero BF16 words and zero
  differences;
- 48/48 shared numerical payloads matching the independent resident reference,
  and zero differences across all 98,304 shared output elements;
- 32 validation cases with nine state/output summaries each, all finite,
  digested and free of poison sentinels;
- 48/48 clean registration and unregister attempt rows.

Its original Markdown report accidentally printed the component pass boolean
as endurance authorization. The canonical JSON says false/closed; a separate
erratum and the 26/26 independent audit preserve the correct interpretation.

## What D10B-R proves

D10B-R freezes a held-out route stream from source partition `[768,1024)` with
SHA-256 `85f12fb0…ffd20`, eight warm-ups, 10,000 measured cases and at most 31
cold records against 32 cold slots. The run records:

- 10,000 finite positive wall and CUDA-event timings;
- wall p95 69.047360 ms and p99 77.218634 ms, below 150/200-ms gates;
- drift ratio 0.870012, below the 1.20 gate;
- 10,000/10,000 finite-and-written composed-state guards;
- 10,000 telemetry rows, minimum RAM 4,564,619,264 bytes and minimum VRAM
  2,844,786,688 bytes;
- 703 page-read samples, maximum 1,020.738/s against the 2,048/s gate;
- exact dense checksum, intact runtime sentinels, and zero CUDA/runner errors;
- 48/48 registrations and 48/48 unregisters without failures.

The frozen checkpoint schedule is step 0 and steps 99, 199, …, 9,999: 101
checkpoints × nine arrays = 909 summaries. Every summary stores shape, dtype,
finite flag, poison count and SHA-256. All are finite and poison-free; all 101
composed-state digests are unique.

Two CPU-only audits pass. The compact audit replays 19/19 evidence checks. The
full provenance/route/resource audit passes 49/49 checks and independently
recomputes all 19 gates. Its 49th check confirms that the active bulk bank
exists at exactly 49,925,652,480 bytes. It validates the manifest's declared
bank SHA-256 but deliberately does not rescan the 49.9-GB payload. A second
count correction supersedes the earlier, incorrect 48/48 erratum.

## Replayability boundary

The checkpoint rows do not store their underlying tensor bytes. Consequently
the CPU audits can verify exact schedule, route metadata, shapes, dtypes,
finite/poison flags, digest format and digest uniqueness, but cannot recompute
the 909 SHA-256 values from tensors. The 10,000 per-step state records are also
Boolean guards rather than raw tensors. D10B-R is therefore a valid pass under
its frozen summary contract, not an independent numerical replay or cross-run
determinism proof.

This distinction is important: component-level exactness is strong in
D10A2-R2, while long-run D10B-R integrity is periodic summary evidence. The
combined result supports sustained synthetic mechanism stability, not full
checkpoint numerical equivalence.

## RAM, paging and cleanup

D10B-R starts with 55,108,874,240 bytes available, has 5,704,151,040 bytes
available after first touch, and never samples below 4,564,619,264 bytes during
the measured run. First-minus-last sampled availability is negative
1,024,204,800 bytes, meaning availability increased rather than showing
monotonic loss. Cleanup leaves 6,875,283,456 bytes available. These observations
pass the preregistered gates, but remain one Windows cache/residency trajectory;
they are not a general OS-reclamation guarantee.

The disk cleanup manifest is retained as evidence. It permanently removed
86,968,817,216 bytes of reproducible superseded bulk artefacts while retaining
the active 49,925,652,480-byte PORT80B backing bank, original checkpoints,
runnable Q5 baseline, reports, manifests, scripts and routes. No result depends
on a silently deleted unique evidence file.

## Honest next gate

The highest-value next experiment is no longer another synthetic cache or
staging variant. It is a new, separately preregistered real-checkpoint
integration gate:

1. acquire an official compatible checkpoint payload and freeze its exact
   commit and tensor hashes;
2. build a differentiated weight bank so wrong layer/expert selection changes
   numerical output, not only headers;
3. capture natural or representative router traces and held-out prompts;
4. implement the actual model shell and compare logits/tokens against an
   independently executable reference;
5. retain enough checkpoint/output bytes to recompute numerical hashes rather
   than only summary digests;
6. measure retained quality, inclusive decode latency/tokens per second,
   RAM/VRAM/page behavior and clean lifecycle over a separately gated endurance
   run.

A pass there would connect the now-strong systems mechanism to an actual model.
A failure would localize the remaining gap to real weight heterogeneity,
model-shell cost, natural routing or quality—not to the already demonstrated
synthetic 499+13 endurance mechanism.

## Strict claim policy

Supported:

- D5 transport, D7 staged-Q5 and D9 capacity-bridge component passes;
- D10A2-R2 exact shape-informed component composition;
- D10B-R sustained held-out synthetic endurance under frozen latency,
  state, paging, RAM, VRAM and cleanup gates;
- two independent CPU audits of D10B-R provenance, arithmetic and stored
  summary evidence;
- immutable preservation of all failed and invalid attempts.

Not supported:

- an official 80B/Next checkpoint run;
- natural router behavior or retained language-model quality;
- production end-to-end tokens per second;
- independent recomputation of the 909 checkpoint digests from raw tensors;
- cross-run determinism, clean 100% bank registration, or a universal OS-memory
  reclamation guarantee;
- an industrial or field-level LLM breakthrough.
