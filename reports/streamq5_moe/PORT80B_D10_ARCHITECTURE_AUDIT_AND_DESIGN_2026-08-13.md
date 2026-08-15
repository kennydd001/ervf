# PORT80B-D10 — independent architecture audit and safe design

**Date:** 2026-08-13  
**Method:** CPU-only source/artifact audit; no GPU execution, bank build, or registry edit.

## Verdict

D10 is **not ready for a defensible 10,000-step GPU run**. D9 supplies a real,
physically measured 499-mapped + 13-pageable expert transport/compute plane,
but there is no ready Qwen3-Coder-Next dense+KV shell to compose with it.
N4B-R's 28.077-ms dense number is a byte projection, not such a shell.

The right next step is a separately preregistered **D10-A short physical
composition**. It must first implement and prove the missing Qwen3-Next
Gated-DeltaNet, specialized full-attention/KV, shared-expert and stateful
layer-composition paths. Only a D10-A correctness and headroom pass may open a
**D10-B 10,240-step endurance run**.

## What can actually be reused

| Existing evidence or code | Reuse in D10 | Boundary |
|---|---|---|
| D9 `host_to_smem_pipeline`, 499+13 cold escape, width-8 staged Q5 kernels, header oracle | Reuse directly for the ten routed records/layer | D9 is a stateless active plane with one fixed input; it does not aggregate routed outputs or carry decoder state |
| P7B/P7C generic Q8 ERVF GEMV | Reuse for official-shape Q8 matrices: attention/DeltaNet projections, routers, output projections and LM head | The official 80B Q8 records do not yet exist; old P6/P7 records are Qwen3-30B weights/shapes |
| P6/P7C RMSNorm, BF16 helpers and residual kernels | Reuse structurally; hidden size 2,048 already matches | Must be rebound to the Qwen3-Next layer graph and exact rounding order |
| P6/P13A/P13C KV, RoPE and EVT-PM attention design | Use as the physical starting point | Current kernels are hardcoded to 32 Q heads, 4 KV heads, head dimension 128, 4,096 context and 48 attention layers. Qwen3-Next needs 16 Q heads, 2 KV heads, head dimension 256, a sigmoid Q gate, and only 12 full-attention layers. They are not valid unchanged |
| N2D exact Q8 LM-head/argmax path | Reuse directly by shape: 151,936 × 2,048 | Requires a new official-shape synthetic or real Q8 LM-head record |
| P6 host embedding lookup | Reuse directly by shape: 151,936 × 2,048 | Existing payload belongs to Qwen3-30B, not Qwen3-Coder-Next |
| N4B-R routed+shared Q5 arithmetic | Reuse as the exact arithmetic oracle/schedule | It computes eleven independent experts, but does not implement shared-expert sigmoid gating, routed weighting, summation and residual composition |
| llama.cpp `ggml-cuda/gated_delta_net.cu` plus `src/models/qwen3next.cpp` | Strong semantic and CUDA implementation source for the missing recurrence | The current local llama build is CPU/Linux-only; this CUDA source is neither exposed to the CuPy runner nor physically measured on this GPU |
| P4D route tensors | Reuse only as a **representative Qwen30-derived traffic trace** | They are top-8 over 128 Qwen3-30B experts, not natural top-10/512 Qwen3-Coder-Next routes |

The old `p6a_exact_runtime_bank` can remain a kernel fixture, but it cannot be
the D10 dense shell. Its architecture and payload are the wrong model. The
N4B-R 1.071-GB active bank is also not required once D10 uses the PORT80B bank.

## Exact missing architecture

N4A verifies the official 80B layout: 48 layers, of which 36 are Gated
DeltaNet and 12 are full attention, followed in every layer by top-10 routed
MoE plus one gated shared expert. A physical D10 token must therefore execute,
in layer order:

1. input RMSNorm;
2. either the complete recurrent DeltaNet state transition or the complete
   Q-gated full-attention/KV update;
3. attention residual;
4. post-attention RMSNorm and 512-way router Q8 GEMV;
5. ten routed D9 Q5 experts, deterministic route weights and weighted sum;
6. one shared Q5 expert, sigmoid shared gate, add and residual;
7. after layer 47, final norm and physical Q8 LM head/argmax.

No current STREAMQ5 runner performs that graph for Qwen3-Next. In particular,
D9 stages 480 records and evaluates every layer against the same input vector;
it never feeds a layer's combined output into the next layer. That stateful
compositor, not a timing-loop extension, is the immediate blocker.

## Physical memory plan at 4K

N4A and P7C support the following conservative device allocation before any
timing:

| Item | Bytes |
|---|---:|
| measured CUDA/context reserve | 1,161,363,456 |
| official-shape Q8 dense core + LM head | 1,933,921,280 |
| resident shared-Q5 bank | 97,320,960 |
| 4K full-attention KV + recurrent/conv state | 178,520,064 |
| D9 staged routed plane | 973,209,600 |
| D9 full-size cold-escape buffer | 973,209,600 |
| frozen extra scratch reserve | 268,435,456 |
| **accounted total** | **5,585,980,416** |

Against the measured 8,546,484,224-byte device this leaves
**2,960,503,808 bytes (2.757 GiB)** before small activations and compiled-module
overhead. D10-A must physically allocate these buffers before registering the
499 prefixes and must record `memGetInfo` after every allocation. Analytical
fit alone is not a gate pass.

Future large artifacts strictly required are:

- one differentiated 49,925,652,480-byte Q5 bank plus manifest (the current
  invariant bank is sufficient only for D9 transport/integrity replay);
- one official-shape Q8 dense/core/LM/embedding bank, approximately
  2,249,948,160 bytes including the host embedding;
- the existing 48 P4D route tensors and their locks (3,964,416 bytes total).

No second N4B-R bank and no deleted P1D/coretail bulk bank is required.

## RAM and page telemetry already available

D9 records only three system-RAM snapshots: 52.887 GB before registration,
52.778 GB after registration and 3.123 GB after first-touch execution plus
clean unregister. It has no time-resolved page telemetry. Therefore D9 does
not establish endurance headroom or recovery.

The reusable P0 telemetry implementation provides:

- 1-Hz Windows PDH `Memory/Page Reads/sec` and `Pages Input/sec`;
- process RSS, peak working set, pagefile/commit, peak pagefile, private bytes
  and total hard+soft page faults;
- system total/available/used RAM and swap;
- GPU temperature, power, clocks and used memory through `nvidia-smi`.

P13C also demonstrates periodic process/GPU telemetry and a 256-MiB pagefile-
growth gate. D10 should combine these. PDH counters are system-wide, so idle
baseline samples and the raw arrays must be retained; they cannot attribute a
page-in uniquely to D10.

## Safe representative routes

There is no natural 80B route artifact. Natural routes require either a real
Qwen3-Coder-Next execution or externally captured, provenance-locked top-10
routes and weights. P4D alone cannot supply them.

A safe interim trace is explicitly named `representative_lifted_p4d`, never
`natural_80b`:

1. preserve P4D's five domains, token order, 48 layers and fixed partitions
   calibration `[0,512)`, validation `[512,768)`, test `[768,1024)`;
2. map each original expert `e` to
   `4*e + SHA256("D10LIFT|domain|layer|e") mod 4`, which keeps a stable mapping
   and therefore preserves the source trace's temporal reuse;
3. form ranks 8 and 9 from ranks 0 and 1 using a different frozen SHA-256 salt
   and one of the three unused sublanes; retry deterministically only on a
   collision;
4. use frozen synthetic rank weights such as FP32 `softmax(-rank/2)` and state
   clearly that neither IDs nor weights are an 80B-router quality result.

A CPU replay of this exact lift found zero duplicate top-10 rows. Cold-tail
IDs 499…511 account for 2.5733% of calibration occurrences, 2.7420% of
validation and **2.7858% of test**, equal to **13.372 of 480 records/token** in
test. These rates are descriptive properties of the frozen lift, not estimates
of natural Qwen3-Coder-Next traffic. D9 all-hot/mixed/all-cold cases should
remain separate mechanism controls.

## Required preregistration gates

### D10-A — short correctness/composition gate

All must pass before endurance is authorized:

1. SHA-256 bind the preregistration, runner, bank/manifests, N4A/N4B-R/D9/P7C/
   P13 evidence, llama.cpp Qwen3-Next/GDN sources, P4D capture/lock and all 48
   route tensors; refuse overwrite and map banks read-only.
2. Use deterministic differentiated Q5 codes and scales. Full-bank SHA and
   sampled header/payload CRCs must match; invariant payload is not sufficient.
3. Independently check Gated DeltaNet, full attention/Q-gate, shared-expert
   gate/add and complete one-layer outputs against CPU references with zero bit
   differences under the frozen BF16/FP32 semantics.
4. On at least 24 fixed tokens, the full 48-layer state, 12-layer KV writes,
   36 recurrent-state writes and final logits/argmax digest must match the
   independent reference. D9 positive image mismatch must be zero and its
   wrong-expert/wrong-layer controls must remain positive.
5. Allocate the complete Q8 shell, 4K KV/recurrent state, shared bank, full D9
   staging and cold escape before registration; require at least 512 MiB actual
   free VRAM afterward and at least 2 GiB available system RAM at every safety
   checkpoint.
6. Physically time components and their serial composition; no projected time
   may enter pass/fail. Dense+KV-shell p95 must be at most 40 ms, routed expert
   p95 at most 65 ms, composed p95 at most 100 ms; strong composed p95 is at
   most 90 ms.
7. Exactly 48 registered ranges, clean reverse unregister, no CUDA/runner/
   driver error, and no result if cleanup fails.

### D10-B — endurance gate, opened only by D10-A

Use the 1,280-token held-out test stream for eight disclosed cycles = 10,240
steps; it is an endurance replay, not 10,240 independent language tokens.
Require:

- finite inclusive wall timings with mean ≤100 ms, p95 ≤120 ms, p99 ≤150 ms
  and ≥10.0 steps/s; strong p95 ≤90 ms;
- identical final hidden/KV/recurrent-state/prediction digests across an
  independent replay and exact state-write counts;
- 1-Hz raw PDH plus process/system/GPU telemetry, with a 60-s idle baseline
  and at least 1,024 untimed warm-up steps;
- post-warm-up `Pages Input/sec` p99 ≤`max(2048, 4× idle p99)`, and at most 1%
  of samples above `max(8192, 8× idle p99)`;
- available RAM never below 2 GiB; last-512-step median no more than 512 MiB
  below the first post-warm-up 512-step median; system pagefile growth ≤256 MiB;
- peak process commit ≤58 GiB, no thermal/driver error, clean unregister, and
  within 120 s after cleanup available RAM returns to within 2 GiB of its
  preregistration value.

## Concrete blocker and decision

The blocker is **not disk space or D9 transport**. It is the absence of an
integrated, exact and physically validated Qwen3-Next layer shell—especially
the 36-layer Gated DeltaNet recurrence, the dimension-correct 12-layer
Q-gated attention/KV path, shared-expert gating, routed aggregation and
stateful layer chaining. A real 80B route stream and route weights are a second
blocker for any natural-traffic or model claim.

Proceed with D10-A only after those kernels and independent oracles are frozen.
Do not call a P7C dummy workload plus D9 timing a Qwen3-Next shell, and do not
add N4B-R's projected 28.077 ms to D9 as if it were physical evidence.
