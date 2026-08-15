# N4_ZERO_CACHE_DATAPLANE — preregistration

Registry: LIGHTNINGSTREAM_NEMOTRON · Phase: `N4_ZERO_CACHE_DATAPLANE` (H3)
Datum: 2026-08-14
Status bij schrijven: **design frozen, execution not yet authorized**
Depends on: `N3_ONE_MODULE_REFERENCE` (PASS, 12/12 gates)
Protected baseline: root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`

## 1. Vraag

Can the routed-expert path for one token be executed **without any expert
cache** — host-resident NVFP4 bank, physical H2D transfer, on-GPU decode and
`up → ReLU² → down` — bit-faithfully against the N3-validated CPU reference, and
within the frozen transfer/latency screen?

Zero cache first is deliberate. It is the cleanest active-set baseline, it cannot
be flattered by a favourable cache policy, and the project's own history is
explicit that a zero-cache pass matters more than a sophisticated cache result.

## 2. Inherited evidence, bound by hash

| source | fact |
|---|---|
| N1 header inventory | 138 routed records/token at top-6; 774,533,280 B/token; all-cold floor 29.608769 ms at 26.158915 GB/s |
| N2 tensor inventory | 2,944 uniform 5,612,560-B records; **3 contiguous ranges per expert** (4,988,928 B codes + 623,616 B scales + 16 B globals); 4 experts straddle a shard boundary |
| N2 decoder validation | NVFP4 decode bit-exact, round trip over 9,977,856 codes |
| N3 nibble resolution | packing order `low_first`, confirmed against torchao 0.18.0 |
| N3 module reference | routed expert rel L2 3.008e-07 vs official; MoE aggregate 1.886e-07 |
| PORT80B DirectPath (protected, read-only) | the 80B zero-cache p50 of 63.034 ms decomposed as 37.204 ms PCIe + 25.830 ms host gather — **dispatch and gather, not PCIe bandwidth** |

The PORT80B diagnosis is the reason this phase measures per-record versus batched
transfer separately instead of reporting one aggregate number. Nemotron moves
774.533 MB in 138 records where Qwen3-Coder-Next moved 973.210 MB in 480, so the
dispatch term should shrink by roughly 3.5× on record count alone. That is a
prediction, recorded before measurement.

## 3. Sub-phases

### N4-A — physical host bank
Build a host-resident bank of the routed experts for a **fixed subset of layers**
in the exact 3-range layout, from the immutable shards. Verify every emitted
record against a fresh read of the checkpoint. No dequantization at build time.

### N4-B — transport
Measure real pinned H2D of one token's routed working set. Three arms:
1. per-record copies (one `copy_` per range);
2. per-layer batched copies (ranges concatenated per layer);
3. single contiguous staged copy of the whole token working set.

### N4-C — on-GPU decode and expert compute
Decode NVFP4 on the GPU and execute `up → ReLU² → down` with the official
weighted top-6 reduction and the shared expert resident on device.

### N4-D — composed zero-cache token path
N4-B + N4-C for one token, end to end, no cache, no reuse between tokens.

## 4. Exactheid — the correctness rule

The GPU path is compared against the **N3-validated CPU reference** on identical
inputs and identical routes.

- Decoded NVFP4 weights on GPU must be **bit-identical** to the CPU decode. This
  is an integer unpack plus exact table lookup, so equality is required, not a
  tolerance.
- Expert output comparison runs in float32 with `rel_l2 <= 1e-5`, matching the
  N3 routed-expert gate. Cross-device float reduction order is not required to
  match, per the assignment's rule against demanding cross-backend bit identity.
- No full dequantized routed bank may be materialised in host or device memory
  at any point. The runner must record peak device allocation and prove the
  working set never exceeds one token's routed set plus fixed residents.

## 5. Frozen gates

| # | gate | threshold |
|---|---|---|
| G1 | every emitted bank record bit-identical to a fresh checkpoint read | exact |
| G2 | GPU-decoded weights bit-identical to CPU decode | exact |
| G3 | expert output vs N3 CPU reference | `rel_l2 <= 1e-5` |
| G4 | routed-expert path p95, full 138-record token, best transport arm | **<= 45 ms** |
| G5 | no full dequantized routed bank materialised | required |
| G6 | peak device allocation recorded and within 8.0 GiB | required |
| G7 | no protected byte changed | required |

G4 is the hard screen from the assignment. Its stop rule: **if p95 exceeds
60 ms after registered/batched transfer and a correct expert compute, reassess
the physical architecture before building a full model.** Between 45 and 60 ms is
a fail of G4 that does not trigger the architectural stop.

Measured latency is reported as mean, p50, p95, p99 and max over the
preregistered repetition count. No projected number may enter a gate.

## 6. Meetprotocol

- Batch 1, one token, greedy path, no cache between tokens.
- Warmups excluded and counted separately; repetitions fixed **before** results
  are opened: 5 warmup, 30 measured per arm.
- CUDA events for device timing, `perf_counter_ns` for wall time; both retained.
- GPU temperature and free memory sampled before and after each arm.
- All raw per-repetition values persisted, not only summaries, so an independent
  verifier can recompute every percentile.
- Routes come from the frozen N3 official capture. They are **synthetic-input
  routes** and every artifact must say so.

## 7. Non-interference

The GPU is shared with the PORT80B/STREAMQ5 agent. Before every GPU arm the
runner checks for a live Python process and current device memory use. If either
indicates the 80B line is active, the GPU arms are **skipped and recorded as
`blocked_gpu_busy`**, the CPU-side sub-phases still run, and the phase is
reported as partial rather than proceeding. Under no circumstance is another
process killed, suspended or reniced.

## 8. Stop rules

- G1 or G2 fails → the bank builder or the GPU decode is wrong. Debug the
  implementation; make **no** statement about the checkpoint.
- G4 fails at or below 60 ms → record the failure, keep the phase open, do not
  tune the gate.
- G4 exceeds 60 ms → architectural reassessment before any full-model work, per
  the assignment stop rule.
- Device OOM → report the actual working set; do not reduce precision to hide it.

## 9. Claim boundary

N4 may claim only: a physically measured, cache-free routed-expert dataplane for
one token on this specific GPU, with bit-exact decode and reference-matching
output. It may **not** claim tokens per second, full-model latency, quality,
memory feasibility of the complete runtime, or any comparison with other
runtimes. A component measurement may never be promoted to tok/s — that rule
killed CORETAIL and applies here unchanged.

## 10. Artefacten

| path | kind |
|---|---|
| `scripts/lightningstream_nemotron/n4_zero_cache_dataplane.py` | runner |
| `scripts/lightningstream_nemotron/n4_independent_verify.py` | independent verifier |
| `reports/lightningstream_nemotron/n4_zero_cache_dataplane.json` | raw result incl. per-repetition arrays |
| `reports/lightningstream_nemotron/n4_independent_verification.json` | verifier output |
| `reports/lightningstream_nemotron/N4_ZERO_CACHE_DATAPLANE_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n4_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n4.json` | protected check |

The independent verifier is a **separate program**. It re-reads the raw arrays
and recomputes hashes, gates, percentiles and byte accounting without importing
the runner, and reports an explicit N/N check count.
