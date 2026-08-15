# PORT80B-D10A1-R failure counter-audit

Date: 2026-08-13  
Mode: CPU/source-only; no GPU execution; no registry mutation.

## Verdict

D10A1-R remains an immutable negative result and does not authorize endurance.
However, its three failed component gates do **not** currently falsify the
component mechanisms. All three are best explained by one deterministic source
defect: host/device initialization and verifier uploads are submitted on the
CuPy current/default stream, while their consumers are launched immediately on
a separately created `cp.cuda.Stream(non_blocking=True)`. A non-blocking stream
has no guaranteed dependency on those default-stream operations.

The header failure has high-confidence evidence of this race. The GDN-conv and
shared-Q5 failures have the same source hazard and failure signature. Kernel
geometry and shared-record addressing check out arithmetically, so there is no
positive source evidence for a Q5 layout or GDN formula defect. A new,
preregistered D10A2 single-stream repair is scientifically justified and is the
minimal falsification test. The prior negative must not be overwritten.

## Immutable evidence checked

- D10A1 component source SHA-256:
  `ffde9c13a3d6d19e3e1132369a4eb9a2e98a4e974bbece86ec224e2931f0ecfd`.
- Imported D2 verifier source SHA-256:
  `20ce962d3196095354688f3bc6cdc378c6351d0dc21f054a57ad46dec5f7d616`.
- D10A1-R raw JSON SHA-256:
  `c92e5dda380c8f9ed0669fc8961056bef58fabbf758946776b426fa7feb888ae`.
- P0 bank manifest SHA-256:
  `466250b39c5f28c8b5b2f7432904100adf8c9e8c246c860a233cdf4b9aeb412c`.

Raw status is `component_composition_negative_endurance_closed`. Exactly 18 of
21 gates passed. The three false gates were:

1. `all_correctness_headers_zero_mismatch`;
2. `gdn_reference_abs_rel_le_2e_5`;
3. `shared_q5_bitexact`.

The same run nevertheless passed all raw canary checks, all 40 routed-Q5 exact
comparisons, output-digest diversity, differentiated wrong-expert and
wrong-layer controls, attention, dense/runtime touch, all resource gates, the
32 validation timings, and clean unregister. Validation wall p50/p95/p99 was
78.771/87.452/90.158 ms.

## Root cause 1: header verifier crosses streams

The component runner creates `stream = cp.cuda.Stream(non_blocking=True)` at
line 646. The imported `full_verify()` creates both its expected header array
with `cp.asarray(...)` and its mismatch scalar with `cp.zeros(...)` without a
`with stream:` context, then launches `verify_record_bytes` on `stream`. Thus
the kernel can read an incompletely uploaded expected array. The final
`stream.synchronize()` only waits for the consumer stream; it does not create a
prior producer dependency.

The evidence strongly isolates the verifier rather than staged bytes:

- 36/40 cases reported zero header mismatches;
- only cases 12, 14, 15 and 28 failed, with 151, 156, 73 and 111 mismatches,
  respectively (491 total);
- every raw header-derived expert ID and numerical canary was exact in all 40
  cases;
- every candidate/oracle Q5 output was bitexact in all 40 cases.

If staged headers were genuinely wrong, the independent header-derived IDs and
canaries should fail coherently. Sparse, non-repeatable-looking verifier counts
alongside exact staged data are the characteristic signature of the expected
header upload race. The default-stream zeroing of `mismatches` is also unsafe
and can mask rather than create some counts.

Minimal repair: make verifier buffers and copies on the same stream as the
kernel. Prefer a local verifier that preallocates expected headers and the
mismatch scalar inside `with stream:`, clears the scalar inside the same
context, launches the kernel, synchronizes, and only then copies the scalar to
host. Do not reuse the current cross-stream `full_verify()` unchanged.

## Root cause 2: GDN conv output can be zeroed after the kernel

Immediately before the component kernels, line 844 calls
`attention.fill(0); recurrent.fill(0); conv.fill(0); ...` on the current/default
stream. The GDN kernel is then launched on the non-blocking component stream.
Consequently `conv.fill(0)` may execute after `gated_deltanet_step` writes the
conv buffer. The raw result `conv_nonzero = 0` is exactly this failure mode.

The kernel geometry itself is correct:

- `CONV_BYTES / 2 = 36 * 8192 * 4 = 1,179,648` BF16 words;
- the launch has `RECURRENT_BYTES / 4 = 18,874,368` logical threads, so all conv
  indices are covered by the kernel's second branch;
- at step zero only slot zero is written;
- 64 of every 8,192 channel values are exactly zero, so the deterministic
  expected nonzero count is `36 * (8192 - 64) = 292,608`.

The current gate merely asks for `conv_nonzero > 0`; D10A2 should freeze the
stronger exact value 292,608 at step zero and record a BF16-word digest or an
exact CPU-derived sample. Initialization and all later resets at line 890 must
be queued on the component stream before their consuming kernels.

## Root cause 3: shared-Q5 output has the same reset race

`shared_gate`, `shared_up` and `shared_down` are also cleared by default-stream
`fill(0)` calls at line 844 immediately before the three shared kernels are
launched on the non-blocking component stream. In particular,
`shared_down.fill(0)` can overwrite computed rows. The observed 96,256 differing
elements out of 98,304 (maximum absolute error 0.189453125) is consistent with
a partially ordered clear, not proof of a Q5-addressing defect.

The static layout audit found no addressing error:

- the P0 manifest fixes layer-major order, experts 0..512, then gate/up/down;
- each shared record is copied to `shared + layer * 2,027,520`;
- gate/up/down matrix strides are 675,840 bytes;
- each matrix addresses 64 header bytes, 655,360 Q5 code bytes and 16,384 BF16
  scale bytes;
- gate/up dispatch covers exactly 48 * 1,024 rows and maps each layer to two
  512-row projections;
- down dispatch covers exactly 48 * 2,048 rows and uses the matching
  `activation + layer * 512` slice;
- the retained synthetic bank manifest explicitly fixes all numerical payloads
  to code byte `0x55` and BF16 scale word `0x3c00`, so comparing every layer's
  shared expert with the resident layer-0 expert is valid for this synthetic
  bank. Headers differ, but the compute kernels deliberately begin after the
  64-byte header.

The 2,048 equal elements do not rescue the run and cannot prove which clear
interleaving occurred. They only reinforce that a rerun must make ordering
deterministic. D10A2 should additionally verify payload equivalence excluding
headers and compare all `48 * 2,048` output BF16/FP32 bit patterns.

## Wider ordering hazards that must be repaired together

Fixing only the three visible sites would leave latent flakiness. The same
producer/consumer pattern also appears in:

- initial `x = cp.asarray(x_host)` followed by component-stream kernels;
- `pointer_table()` returning a default-stream `cp.asarray(...)` immediately
  consumed by the staging kernel;
- `assemble_oracle()` creating `route_device` on the default stream immediately
  consumed by `differentiate_q5_expected`;
- `canary_errors.fill(0)` before a component-stream verifier;
- `dense_checksum.fill(0)` before component-stream dense work;
- the recurrent, conv, KV and checksum resets before validation.

Therefore the deterministic repair must establish one stream ownership rule
for **all** allocations whose initialization is consumed, all host-to-device
uploads, fills, pointer/route arrays, kernels and event timing. The simplest
robust implementation is to create the non-blocking stream and perform all
CuPy allocations/uploads/fills within `with stream:`; explicit raw-runtime
copies already receive `stream.ptr`. Synchronize before every host read. A
source-local helper for same-stream zero/upload would make the contract
auditable.

## D10A2 preregistration recommendation

A new preregistration is required because correcting dependency ordering changes
experimental code after observing a negative. D10A2 should be narrowly named
and frozen as a *single-stream deterministic repair*, with this mutation
whitelist:

1. move all CuPy allocations, initialization, uploads, clears and verifier
   buffers onto the one experiment stream, or add explicit event dependencies;
2. replace imported `full_verify()` with a same-stream equivalent;
3. strengthen the step-zero conv oracle to exactly 292,608 nonzero BF16 words
   plus a frozen digest/sample;
4. add a header-excluded shared-payload equivalence check and retain the exact
   98,304-element shared output comparison;
5. initialize output buffers to distinct sentinels on the same stream and assert
   that no required output retains its sentinel.

Everything else should remain frozen: bank, route cases, 40 correctness and 32
validation cases, raw canary arrays, negative controls, Q5 math, resource gates,
timing limits, page-read/RAM/VRAM gates and claim boundary. No threshold tuning
or route selection is justified. D10A1-R stays negative and immutable.

The independent verifier should check source hashes and the mutation whitelist,
recompute all booleans from raw stored arrays, confirm the exact conv oracle,
recompute shared bit comparisons and digests, and assert that every relevant
producer and consumer is on the same stream. It must not infer a pass from the
absence of a CUDA error.

## Decision boundary

- **D10A2 is justified:** yes, as a falsifiable repair of an identified
  synchronization defect.
- **Can D10A1-R be reclassified now:** no; it remains a valid negative execution
  of flawed experimental plumbing.
- **Is Q5 addressing proven correct:** statically consistent, not dynamically
  proven by this failed run. A clean D10A2 rerun is needed.
- **Is endurance authorized:** no. Only a clean component pass plus independent
  replay can reopen a separately preregistered endurance phase.

