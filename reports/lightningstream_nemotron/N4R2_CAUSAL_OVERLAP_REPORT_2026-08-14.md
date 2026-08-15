# N4-R2 — causal H2D/compute overlap

Datum: 2026-08-14
Verdict: **PASS. Zero-cache routed path 39.714 ms p95 under a 45 ms screen, bit-identical to the serial path.**
Terminal state: `n4r2_overlapped_zero_cache_screen_pass`
Independent verification: **34/34**

## Kernresultaat

The H3 zero-cache screen is met. Overlapping each layer's transfer with the
previous layer's compute closes the 2.421 ms R1 gap and changes **zero output
bits**.

| arm | p50 | p95 |
|---|---:|---:|
| transport only | 29.888 ms | 30.017 ms |
| fused compute only | 18.454 ms | 37.828 ms |
| composed serial | 45.853 ms | 50.917 ms |
| **composed overlapped** | **31.506 ms** | **39.714 ms** |

Cumulative against the frozen baselines:

| baseline | p50 | ratio |
|---|---:|---:|
| N4 unfused composed | 376.244 ms | **11.94×** |
| N4-R1 fused serial | 46.954 ms | **1.490×** |

## De beslissende poort: O3

Overlap is only admissible if it is semantically invisible. Measured on the same
data in the same run:

```text
differing words     0 of 2,688
serial   SHA-256    matches
overlap  SHA-256    matches
```

**Zero differing bits.** A speedup with even one changed bit would have been a
failure, not a trade-off. This reproduces on Nemotron what STREAMQ5 P4A
established on the protected line: causal same-layer overlap with bit-identical
end states. The mechanism was ported, not invented.

## Exactheid

| check | result |
|---|---|
| fused expert output vs N3 CPU reference | `rel_l2 = 1.654e-07`, gate 1e-5 |
| fused activation vs reference | `rel_l2 = 1.539e-07`, gate 1e-5 |
| overlapped vs serial accumulator | **bit-identical, 0/2,688 words differ** |
| all outputs finite | yes |
| peak device pool | 774,601,728 B, i.e. working set + 68,448 B |

Bit identity is claimed **only** between the overlapped and serial GPU paths.
Against the numpy reference the fused kernel uses a different reduction order,
so only `rel_l2` is claimed there. Conflating the two would be an overreach.

## Mechanism

Two CUDA streams over the resident device bank. `copy_stream` issues layer `L`'s
three ranges and records event `E[L]`; `compute_stream` waits on `E[L]` before
launching that layer's twelve kernels. Causality is enforced by the events, so
no kernel can read a range before its copy completes — which is exactly why O3
is the gate that matters: a missing event would show up as changed numbers, not
as a crash.

The prediction recorded before measurement was "roughly 33 ms", reasoning that
with compute shorter than transport a fully pipelined token approaches transport
plus one layer's compute. Measured p50 is 31.506 ms against transport-only
29.888 ms — the pipeline is close to transport-bound, as predicted.

## Meetprotocol en variantie

- CuPy 14.1.1, NVRTC, compute capability 120; 5 warmup + 30 measured, fixed before results were opened.
- Working set 138 records × 5,612,560 B = 774,533,280 B, identical to N4 and N4-R1.
- Serial and overlapped paths measured in the same run on the same data, so the comparison and the O3 bit-check are like for like.
- Non-interference: 0 foreign CUDA contexts.

**Variance is the honest weak point of this result.** The fused-compute arm
measured p50 18.454 ms against p95 37.828 ms, and the composed overlapped arm
p50 31.506 ms against p95 39.714 ms. Between R1 and R2 the same compute arm moved
13.762 → 18.454 ms p50, a 34% run-to-run shift on identical work.

The gate is met on p95, but with **5.286 ms of margin**, and the tail is wide
enough that this must not be treated as a settled number. The cause is not
diagnosed here — clock/thermal behaviour on a laptop GPU is the obvious
candidate, but naming it without measuring it is exactly the error this project
recorded over the "glue" term. A thermal and steady-state characterisation is
`N12` work and the margin should be re-measured there.

## Onafhankelijke verificatie

A separate verifier recomputed every percentile from the retained arrays,
re-derived the byte accounting, independently recomputed the routed-expert
reference on the CPU, re-checked the overlap equivalence digests and word
counts, tested the peak-allocation argument against a materialised-matrix
hypothesis, and re-evaluated every gate and the architectural-stop logic. It
imports nothing from the runner and opens no GPU.

Result: **34/34 verification checks passed.**

## Gates

| # | gate | threshold | result |
|---|---|---|:--:|
| O1 | expert output vs reference | `rel_l2 ≤ 1e-5` | ✅ 1.654e-07 |
| O2 | composed overlapped p95 | ≤ 45 ms | ✅ **39.714 ms** |
| O3 | overlapped ≡ serial | exact | ✅ **0 differing words** |
| O4 | peak device, no materialised matrix | ≤ 8.0 GiB | ✅ +68,448 B |
| O5 | outputs finite | required | ✅ |
| O6 | protected bytes unchanged | required | ✅ |

Architectural stop: **not triggered**. Its precondition (a correct fused kernel)
is satisfied, and the measured 39.714 ms is below the 60 ms threshold.

## Eerlijk verdict

What the H3 sequence now establishes, end to end:

1. **Transport is at the roofline.** 774,533,280 B in 29.888 ms; N4 measured 26.03 GB/s against an assumed 26.158915 GB/s.
2. **Pre-pinning eliminates the host-gather term** that cost the protected 80B line 25.830 ms per token.
3. **Batching is not the lever**: 414 copies versus 3 differ by 3.98%.
4. **Fusing decode into the GEMV removes 96.1% of the decode cost** and adds 68,448 B of device memory.
5. **Causal overlap closes the remaining gap and is bit-identical**, 0 of 2,688 words changed.
6. The complete cache-free routed path for one token is **31.506 ms p50 / 39.714 ms p95**, inside the 45 ms screen, with a 0.721 GiB device footprint and **no expert cache of any kind**.

What this does **not** establish: tokens per second, full-model latency,
quality, memory feasibility of the complete runtime, thermal steady state, or
any comparison with another runtime. The routed-expert path is one component;
attention, Mamba-2, the trunk, the LM head and sampling are not in this number,
and a component measurement is never promoted to tok/s — the rule that closed
CORETAIL applies unchanged.

Two honest caveats carried forward: the p95 margin is 5.286 ms with a wide tail
(§Meetprotocol), and the routes are synthetic-input routes from the frozen N3
capture, not natural routes — they determine which bytes move, not how many, so
they are adequate for transport but are not a routing result.

## Vervolg

H3 is satisfied and H5 cache work is now unblocked — a zero-cache pass was the
precondition. The more valuable next step is `N5_PHYSICAL_RESIDENT_SHELL`,
because the measured numbers make the memory question concrete: the routed path
needs 0.721 GiB transient, while N2 measured 2.008 GiB of incompressible BF16 of
which `lm_head` and `embeddings` are 1.312 GiB, and N3 projected 47.078 MiB of
constant Mamba state plus 384 MiB of FP8 KV at 128K.

A cache is worth designing only after the resident shell is physically
allocated, since the cache gets whatever VRAM is left.

## Artefacten

- Preregistratie: `reports/lightningstream_nemotron/N4R2_CAUSAL_OVERLAP_PREREGISTRATION_2026-08-14.md`
- Kernels: `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py`
- Runner: `scripts/lightningstream_nemotron/n4r2_overlap_dataplane.py`
- Machine-readable result: `reports/lightningstream_nemotron/n4r2_overlap_dataplane.json`
- Independent verifier: `scripts/lightningstream_nemotron/n4r1_independent_verify.py`
- Verification output: `reports/lightningstream_nemotron/n4r2_independent_verification.json`
- Voorganger: `reports/lightningstream_nemotron/N4R1_FUSED_NVFP4_KERNEL_REPORT_2026-08-14.md`
