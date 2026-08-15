# PORT80B-D10A — Next component/composition gate preregistration

**Frozen before compile/preflight and before any physical GPU execution:** 2026-08-13

## Why D10A is a gate, not yet an endurance result

D9 proved an exact synthetic 499-mapped + 13-pageable routed-expert bridge,
but did not physically compose the Qwen3-Coder-Next shell. The locally existing
P13 attention kernels implement Qwen3-30B geometry (32 Q heads, 4 KV heads,
head dimension 128, 48 attention layers) and are not a valid Next oracle.
N4BR physically measured the expert plane but only projected dense-shell time.
There is no pre-existing executable implementation of Next's 36 Gated
DeltaNet layers. Therefore D10A first builds and adjudicates a synthetic,
shape-faithful component/composition harness. A 10,000-step endurance phase is
closed unless every component, composition, route, resource and cleanup gate
passes in a separate component run.

## Immutable inputs and bulk-file policy

Required existing bulk data is limited to:

- `reports/runs/streamq5_moe/port80b_p0/port80b_p0_full_q5_bank.bin`, exactly
  49,925,652,480 bytes, read-only, with manifest bank SHA-256
  `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`.

Small required evidence consists of the bank manifest, D9 result and verifier,
N4A shape result, P4D route capture/lock and its 48 small route tensors. No P1D
bank, P6A Q8 payload bank, checkpoint shard, download, generated dense-bank
sidecar or new bulk file is permitted. Synthetic dense weights, shared records,
KV, recurrent state and runtime workspaces exist only as device allocations.
All output paths are new and the runner refuses overwrite. No central registry
is edited.

## Frozen architecture contract

- 48 layers, hidden width 2,048;
- layers 3, 7, …, 47: 12 full-attention layers;
- remaining 36 layers: Gated DeltaNet;
- full attention: 16 Q heads, 2 KV heads, head dimension 256, Q gate present;
- Gated DeltaNet shape: 16 K heads × 128, 32 V heads × 128, convolution width
  4, FP32 recurrent state and BF16 convolution state;
- 512 routed experts, top-10, plus one always-active shared expert per layer;
- routed/shared expert width 512 and aligned record size 2,027,520 bytes.

These are official-shape-derived synthetic components. Without official Next
weight payloads and hidden-state captures, exactness means agreement with the
independently specified D10A numerical reference, not checkpoint equivalence or
model quality.

## P4D-shaped synthetic proxy route contract

No local natural 512-expert Next route traces exist, and P4D came from a
different Qwen3-30B MoE. D10A must therefore label every route
`p4d_shaped_synthetic_proxy`, never `natural`, `representative`, or a Next
trace. It deterministically lifts each
locked P4D Qwen3-30B top-8 expert `e` (0…127) to one expert in the contiguous
Next locality bucket `4e…4e+3`; the lane is selected by frozen SplitMix64 over
domain, source token, layer, rank and repetition epoch. Rank order and coarse
expert locality are preserved. Two independently generated, unique experts are
then appended to form top-10. No route-frequency or quality inference is
allowed.

Source partitions are disjoint:

- correctness: P4D calibration positions 0…7 in each of five domains (40
  source cases);
- validation: P4D validation positions 512…575 (320 source cases);
- endurance: P4D test positions 768…1023, domain-major, repeated for eight
  deterministic lift epochs and truncated to exactly 10,000 steps.

The source routes are reused prior evidence, not fresh data. The lift and
extras must be recomputed independently from route tensors and their hashes.

## Differentiated Q5 numerical and route-integrity oracle

The physical bank has route-specific headers but invariant codes/scales. After
staging, a CUDA differentiation kernel reads the **actual** staged `SQ5M`
header and writes a three-word numerical canary into UP-projection rows 0, 1
and 2, group 0. Let `id = 512*layer + expert`. The three digits are
`id mod 32`, `floor(id/32) mod 32`, and `floor(id/1024) mod 32`; digit `d` maps
to the distinct positive BF16 scale word `0x3e80 + 4d`. This radix-32 triple is
injective over all 48×512 layer/expert pairs. Only staged/oracle HBM images are
patched; the mapped bank is never mutated.

The actual canary is derived only from the staged header. The expected canary
is generated independently from the intended route table, never by rereading
the actual header. Raw intended IDs, actual header IDs, expected three-word
canaries and observed three-word canaries must be retained for every audited
record. A CPU preflight exhaustively enumerates all 24,576 pairs, proves unique
triples, round-trips every triple to the same ID, and explicitly verifies the
expert-498/499 hot/cold boundary for every layer. A wrong source record therefore
changes UP rows 0…2 numerically as well as changing the header.

The component gate requires:

1. zero byte/header mismatches before differentiation for all correctness
   routes;
2. all three staged canary words match the independently generated expected
   table and the raw arrays are internally consistent;
3. full routed Q5 gate/up → BF16 SwiGLU → down output is bitexact against a
   separately assembled resident-record oracle for every correctness case;
4. deliberate same-layer/wrong-expert and wrong-layer/same-expert substitutions
   are detected by both header mismatch and non-bitexact numerical output;
5. output digests vary across at least 95% of correctness routes. A single
   common output digest is an automatic failure.

## Required physical component oracles

The component phase must physically allocate and execute all of the following:

- the 1,933,921,280-byte N4A Q8 dense-shell shape buffer, with every byte read
  by layer-class-appropriate projection work each composed step;
- 12-layer BF16 KV storage at 4,096 context with Next 2-KV-head × 256 geometry;
- 36-layer FP32 Gated-DeltaNet recurrent state and BF16 convolution state;
- the complete 48-record shared-Q5 bank (97,320,960 bytes) resident on device;
- routed staging, bounded cold escape, expert outputs and runtime scratch.

For deterministic nontrivial vectors, CUDA results must agree with independent
NumPy float32/BF16-rounding references:

- Next GQA attention including Q-gate, KV write/read and causal reduction;
- one-step Gated-DeltaNet recurrence and convolution-state rotation;
- shared Q5 gate/up, canonical BF16 SwiGLU, down, shared gate, and routed/shared
  composition;
- dense-buffer checksum/work result and state mutation count.

All compared outputs must be finite; absolute/relative tolerances are zero for
integer/header/digest/state-index checks and at most 2e-5/2e-5 only for the
explicit FP32 attention and recurrence references. Q5 paths are bitexact.

## Phases

### `compile`

Read-only contract audit, Python compile, CUDA source compile and symbol
resolution. It performs no host registration, CUDA kernel launch, large device
allocation, bank scan or timing. Its JSON is a mandatory hash lock for later
phases.

### `component`

Requires a separately authorized GPU run. It performs the 499-prefix
registration, physical allocations, route/numerical controls, component
oracles, 8 untimed composed warm-ups and 32 validation steps. It records
inclusive wall and CUDA-event p50/p95/p99, system page reads, process RSS/
working-set/pagefile, system available RAM, VRAM, CUDA errors and cleanup.

Component pass requires all correctness gates above, exact physical byte sizes,
all allocated buffers actually touched, 32 finite validation steps, inclusive
wall p95 ≤150 ms and p99 ≤200 ms, no post-warm-up hard-page-read sample above
2,048 reads/s, no monotonic loss exceeding 1 GiB across validation telemetry,
at least 2 GiB system RAM after first touch, at least 512 MiB free VRAM at peak,
and clean unregister of exactly 48 ranges. A failure closes endurance.

### `endurance`

Requires a separately authorized run, an immutable passing component JSON and
the exact acknowledgement `D10A_10000_AFTER_COMPONENT_PASS`. It executes exactly
10,000 frozen `p4d_shaped_synthetic_proxy` test steps. It stores per-step latency and
compact telemetry every 64 steps, but no bulk output tensor.

Endurance pass requires:

- exactly 10,000 finite inclusive wall samples;
- wall p50 ≤125 ms, p95 ≤150 ms, p99 ≤200 ms;
- route/header/fingerprint audit at every 256th step and Q5 output digest audit
  at every 1,024th step, all passing;
- after the first 512 steps, page-read p95 ≤512 reads/s and no sample >2,048/s;
- no monotonic process-private/system-available memory loss >1 GiB between the
  medians of steps 512…1,535 and 8,976…9,999;
- system available RAM never below 1.5 GiB during execution and ≥2 GiB after
  first touch; free VRAM never below 512 MiB;
- no CUDA error/nonfinite state and clean unregister of all 48 ranges.

## Hard resource stops and estimated envelope

- Refuse component/endurance before registration if system available RAM is
  below 50 GiB; refuse/cleanup immediately if it is below 2 GiB just after
  registration or first touch, or below 1.5 GiB at any later checkpoint.
- Refuse large allocation unless CUDA free memory exceeds the exact requested
  bytes by at least 512 MiB.
- Device envelope is preregistered at ≤4.25 GiB allocations plus 512 MiB free
  reserve on the 8-GiB GPU. Host registration is exactly 48 × 499 records
  (45.227966 GiB).
- Compile/preflight target is under 30 seconds. Component target is under 3
  minutes. The unopened 10k phase is expected to take roughly 15–25 minutes;
  this estimate is not a pass criterion.

Any hard stop writes only a small diagnostic JSON, synchronizes if possible,
frees device pools and attempts reverse-order unregister. Cleanup failure
overrides every earlier pass.

## Claim boundaries

A component pass establishes only a synthetic, shape-informed **physical shell
stress/composition** and numerical-reference pass. The generic kernels in this
harness must not be called an exact Qwen3-Next shell unless all 36 Gated
DeltaNet and all 12 Next-attention/KV/q-gate components have separately passed
their frozen numerical oracles. An endurance pass would establish only
sustained behavior of that synthetic composition on a deterministic P4D-shaped
proxy stream. Neither result is a real Next checkpoint, natural routing,
official Gated-DeltaNet equivalence, model quality, prefill, native 262K
context, energy, production throughput or end-to-end language-model claim.
