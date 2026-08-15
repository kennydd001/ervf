# N4-R1 — fused NVFP4 expert kernel

Registry: LIGHTNINGSTREAM_NEMOTRON · Phase: `N4_R1_FUSED_NVFP4_EXPERT_KERNEL` (H3 repair, H6 method)
Datum: 2026-08-14
Status bij schrijven: **design frozen, execution not yet authorized**
Depends on: `N4_ZERO_CACHE_DATAPLANE` (NEGATIVE on G4, `n4_zero_cache_screen_fail_unfused_decode_dominates`, 32/32 verified)
Protected baseline: root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`

## 1. Vraag

N4 established that transport is solved (29.756 ms, 99.51% of roofline) and that
the unfused decode is 93.9% of the composed token. This phase asks one question:

> Does a fused decode+GEMV kernel — one that never materialises the dequantised
> matrix — bring the composed zero-cache routed path under the 45 ms screen,
> without changing model semantics?

N4-R1 is a **repair of an implementation**, not a new hypothesis. If it fails,
the architectural stop of N4 becomes reachable for the first time, because its
precondition ("a correct fused kernel") will finally be satisfied.

## 2. Mechanism and the number to beat

The unfused path decodes a matrix into 9,977,856 float32 weights — about 40 MB
written then 40 MB read — to feed a GEMV that consumes each weight once. The
fused kernel assigns one block per output row, streams that row's packed bytes
and block scales from global memory, decodes in registers and accumulates.

Predicted device traffic per matrix, recorded before measurement:

| path | per-matrix traffic |
|---|---:|
| unfused | ~120 MB (2.8 MB read + 40 MB write + 40 MB read + intermediates) |
| fused | **2,806,272 B** (2,494,464 codes + 311,808 scales) |

Target, from N4: decode must fall from ~353 ms to **under ~15 ms** for the
composed path to meet 45 ms alongside 29.8 ms of transport.

## 3. Semantics — unchanged, and how that is enforced

The kernel performs exactly the arithmetic N3 validated: `up → ReLU² → down`,
group-16 block scales along the contraction dimension, FP32 global scale, no
bias. The *decode* is the same integer unpack and table lookup that N4 proved
bit-identical to the CPU float32 decode.

What does change is **reduction order**: the fused kernel reduces with a
block-parallel warp-shuffle tree, not the sequential order of the numpy
reference. Per the assignment, cross-backend bit identity is explicitly not
demanded. Therefore:

- the fused output is gated on `rel_l2` against the N3-validated reference;
- bit-identity is **not** claimed for the fused output and must not be reported as such.

This is the ERVF distinction the project already owns: preserving a chosen
reduction DAG is a stronger property than what is claimed here, and claiming it
without implementing it would be exactly the kind of overreach the registry
forbids.

## 4. Frozen gates

| # | gate | threshold |
|---|---|---|
| F1 | fused expert output vs N3 CPU reference | `rel_l2 <= 1e-5` |
| F2 | fused intermediate activation vs reference | `rel_l2 <= 1e-5` |
| F3 | **composed zero-cache token p95** | **<= 45 ms** |
| F4 | peak device allocation | <= 8.0 GiB |
| F5 | no dequantised matrix materialised at any point | required |
| F6 | all outputs finite | required |
| F7 | no protected byte changed | required |

F5 is checked structurally: the runner allocates no `[rows, cols]` float32
weight buffer, and peak device allocation is reported against the sum of the
bank buffers plus the small activation vectors. A fused path that secretly
materialised a matrix would show up as a ≥40 MB step in peak allocation.

Stop rule inherited from N4, now with its precondition satisfied: **if the
composed p95 still exceeds 60 ms with a correct fused kernel, the architectural
reassessment is triggered.**

## 5. Meetprotocol

- Batch 1, one token, zero cache, nothing reused between tokens.
- Repetition counts fixed before results are opened: 5 warmup + 30 measured.
- Three arms measured separately: transport only, fused compute only (weights
  already resident), composed token.
- The arms are reported as measurements, not as an additive decomposition. Any
  residual is left unnamed unless separately measured.
- Wall time via `perf_counter_ns`, all per-repetition raw arrays retained.
- Working set 138 records × 5,612,560 B = 774,533,280 B, identical to N4 so the
  comparison is like for like.
- Route weights come from the frozen N3 official capture (synthetic-input
  routes). They do not affect timing and must never be described as natural.

## 6. Non-interference

The corrected N4 rule applies: blocked when another PID holds a CUDA context per
`nvidia-smi --query-compute-apps`, or device memory in use exceeds 256 MiB;
fails closed if the query errors. Python process names are context only. No
process is ever killed, suspended or reniced.

## 7. Claim boundary

N4-R1 may claim only: a physically measured, cache-free routed-expert path for
one token on this specific GPU, with reference-matching output within a declared
tolerance. It may **not** claim tokens per second, full-model latency, quality,
memory feasibility of the complete runtime, bit-exactness of the fused output,
kernel novelty, or any comparison with another runtime. A component measurement
is never promoted to tok/s.

## 8. Artefacten

| path | kind |
|---|---|
| `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py` | fused kernels |
| `scripts/lightningstream_nemotron/n4r1_fused_smoke.py` | correctness smoke test |
| `scripts/lightningstream_nemotron/n4r1_fused_dataplane.py` | runner |
| `scripts/lightningstream_nemotron/n4r1_independent_verify.py` | independent verifier |
| `reports/lightningstream_nemotron/n4r1_fused_dataplane.json` | raw result incl. per-repetition arrays |
| `reports/lightningstream_nemotron/n4r1_independent_verification.json` | verifier output |
| `reports/lightningstream_nemotron/N4R1_FUSED_NVFP4_KERNEL_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n4r1_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n4r1.json` | protected check |
