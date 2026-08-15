# PORT80B-D10A2-R2 independent component audit

**Verdict:** **component pass independently verified**. All **28/28** frozen component gates replay true, and 26/26 independent audit checks pass. This does **not** actually open or run endurance: D10A2-R2's canonical JSON says `endurance_authorized_by_evidence=false`, the endurance artifact does not exist, and its runner raises unconditionally. The pass makes a separately preregistered and separately authorized endurance arm evidence-eligible.

## Exactness and component controls

- 40/40 P4D-derived correctness routes independently reconstructed; 40/40 zero header mismatch, exact raw IDs/canaries, finite fully written outputs and candidate/oracle bit equality.
- 40 distinct correctness output digests; every stored output digest matches both comparison-side digests.
- Wrong-expert and wrong-layer controls each change exactly the intended record. Both produce 3 header-byte mismatches and 2,050 differing FP32 output words.
- Full conv oracle: **1,179,648 BF16 words**, **292,608 nonzero**, zero differing words, SHA-256 `cedf5736557919b023d6f7cce73d0064df07236ff1e18b5d8b3fec49d658fa1e`. The unit-test result and preflight bind the current runner/preregistration hashes.
- All 48 shared numerical payloads were independently reread and rehashed excluding their headers; every projection matches the layer-0/expert-0 reference. The stored 98,304-element shared-output comparison is bitexact.
- Attention and sampled recurrent max-absolute error are both 0; all component poison counts are 0; dense checksum and runtime sentinels match.

## Validation and resources

- Wall p50/p95/p99: **80.862050 / 91.204895 / 91.971097 ms**.
- CUDA-event p50/p95/p99: **80.711601 / 91.009719 / 91.828439 ms**.
- Exactly 32 validation rows, each with nine expected shape/dtype/finite/poison/digest summaries. All routed, attention, delta, recurrent, conv and composed digests vary across all 32 steps; shared output is intentionally constant; KV has 25 distinct cumulative-state digests.
- Start RAM exceeded the 52,652,163,072-byte gate by 2.306499 GiB. RAM after first touch was 4.811619 GiB; validation endpoint loss was 86.863 MiB. Minimum validation free VRAM was 2713.000 MiB.
- 11 page-read samples, maximum **130.445/s**, below 2,048/s.

The validation artifact stores only digest summaries rather than underlying arrays. Their presence, format, shape, finiteness and step sequence are independently checked, but those validation SHA-256 values cannot be regenerated from the JSON alone.

## Registration and cleanup

Exactly 48 registration rows and 48 unregister rows are retained. Layers are 0..47 exactly once, every row is attempted and successful with no error, per-range bytes are 1,011,732,480, host pointers follow the exact 1,040,117,760-byte bank layer stride, unregister pointers match registration pointers, and the failure list is empty.

## Endurance decision and report defect

The generated Markdown report is internally wrong: it prints `Endurance evidence-authorized: True` by interpolating `overall`, while the canonical result field is `false`. The status, preregistration and executable runner all keep endurance closed. Therefore:

- **component mechanism gate:** passed;
- **eligibility to design/preregister the next endurance arm:** yes;
- **current D10A2-R2 endurance gate actually open:** no;
- **endurance evidence produced:** none.

Claim boundary: Synthetic shape-informed physical component/composition evidence on P4D-shaped proxy routes and uniform synthetic Q5 payloads; not an official checkpoint, natural routing, model quality, production throughput or endurance result.
