# N4-R2 — causal H2D/compute overlap

Registry: LIGHTNINGSTREAM_NEMOTRON · Phase: `N4_R2_CAUSAL_OVERLAP`
Datum: 2026-08-14
Status bij schrijven: **design frozen, execution not yet authorized**
Depends on: `N4_R1_FUSED_NVFP4_EXPERT_KERNEL` (NEGATIVE on F3 at 47.421 ms vs 45 ms, 28/28 verified)
Protected baseline: root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`

## 1. Vraag

N4-R1 left the composed token at 46.954 ms p50 / 47.421 ms p95, against a 45 ms
screen, with transport 32.273 ms and fused compute 13.762 ms measured
separately. Those two run **serially** in R1: each layer's six records are
transferred, then computed, then the next layer starts.

> Does causal H2D/compute overlap — prefetching layer `L+1` while layer `L`
> computes — bring the composed path under 45 ms **without changing a single
> output bit**?

This is not a new mechanism. STREAMQ5 P4A established causal same-layer overlap
on the protected line with exactly the same-LRU-misses, same-record-bytes and
**bit-identical end states**, at a 1.402× test speedup. It is listed in the
research handoff as a reusable mechanism. N4-R2 ports it, it does not invent it.

## 2. Why this is a legitimate revision and not post-hoc tuning

R1 failed a preregistered gate by 2.421 ms. The temptation is to widen the gate;
that is forbidden. The alternative is a **new preregistered attempt with a named
mechanism, its own gates, and an added correctness requirement stricter than
R1's** — which is what this is.

Overlap is not free of risk: pipelining a transfer against compute that reads
the same buffers is a classic source of races that show up as wrong numbers, not
as crashes. Hence gate O3 below.

## 3. Mechanism

Two CUDA streams over the existing device bank:

- `copy_stream` issues layer `L`'s three ranges and records event `E[L]`;
- `compute_stream` waits on `E[L]` before launching layer `L`'s twelve kernels.

The device bank already holds all 138 records, so no double buffering is needed
for space; the pipelining is purely in issue order. Causality is enforced by the
events: no kernel may read a range before its copy has completed.

Predicted composed time, recorded before measurement: with compute (13.762 ms)
shorter than transport (32.273 ms), a fully pipelined token should approach
transport plus one layer's compute, roughly **33 ms**. If the measurement lands
far from that, the overlap is not working and must be diagnosed rather than
accepted.

## 4. Frozen gates

| # | gate | threshold |
|---|---|---|
| O1 | fused expert output vs N3 CPU reference | `rel_l2 <= 1e-5` |
| O2 | **composed overlapped token p95** | **<= 45 ms** |
| O3 | **overlapped accumulator bit-identical to the serial accumulator** | **exact, 0 differing bits** |
| O4 | peak device allocation | <= 8.0 GiB and no materialised matrix |
| O5 | all outputs finite | required |
| O6 | no protected byte changed | required |

O3 is the gate that matters most. Overlap is only admissible if it is
semantically invisible. A speedup with even one changed bit is a **failure**,
not a trade-off, and must be reported as such.

Inherited stop rule, precondition now satisfied: if the composed p95 still
exceeds 60 ms with a correct fused kernel, the architectural reassessment is
triggered. R1 already measured 47.421 ms, so this is not expected to fire; it
remains in force.

## 5. Meetprotocol

- Identical working set to N4 and N4-R1: 138 records × 5,612,560 B = 774,533,280 B.
- 5 warmup + 30 measured repetitions, fixed before results are opened.
- Both the serial and the overlapped composed path are measured in the same run,
  on the same data, so the comparison and the O3 bit-check are like for like.
- Wall time via `perf_counter_ns`; all raw per-repetition arrays retained.
- The arms are measurements, not an additive decomposition. Any residual stays
  unnamed unless separately measured.

## 6. Non-interference

The corrected rule from N4 applies unchanged: blocked when another PID holds a
CUDA context per `nvidia-smi --query-compute-apps`, or device memory in use
exceeds 256 MiB; fails closed on query error. No process is ever killed,
suspended or reniced.

## 7. Claim boundary

N4-R2 may claim only: a physically measured, cache-free, causally overlapped
routed-expert path for one token on this specific GPU, output-identical to the
serial path and reference-matching within a declared tolerance. It may **not**
claim tokens per second, full-model latency, quality, memory feasibility of the
complete runtime, bit-exactness of the fused output against the numpy reference,
kernel novelty, or any cross-runtime comparison.

## 8. Artefacten

| path | kind |
|---|---|
| `scripts/lightningstream_nemotron/n4r2_overlap_dataplane.py` | runner |
| `scripts/lightningstream_nemotron/n4r1_independent_verify.py` | independent verifier (shared) |
| `reports/lightningstream_nemotron/n4r2_overlap_dataplane.json` | raw result |
| `reports/lightningstream_nemotron/n4r2_independent_verification.json` | verifier output |
| `reports/lightningstream_nemotron/N4R2_CAUSAL_OVERLAP_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n4r2_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n4r2.json` | protected check |
