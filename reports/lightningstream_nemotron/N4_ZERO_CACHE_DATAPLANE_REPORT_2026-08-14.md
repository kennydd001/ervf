# N4 — zero-cache routed-expert dataplane

Datum: 2026-08-14
Verdict: **G4 NEGATIVE with an unfused decode; transport PASSES at the roofline. Architectural stop NOT declared — its precondition is unmet.**
Terminal state: `n4_zero_cache_screen_fail_unfused_decode_dominates`
Independent verification: **32/32**

## Kernresultaat

The bottleneck is not where the 80B line found it. Transport for a full
138-record token lands essentially on the PCIe roofline, while the **unfused
NVFP4 decode costs 93.9% of the composed token**.

| stage | p50 | p95 | note |
|---|---:|---:|---|
| transport, best arm | **29.756 ms** | **29.848 ms** | 26.03 GB/s |
| decode only | 353.133 ms | 361.664 ms | upper bound, see §Meetprotocol |
| composed token | 376.244 ms | 403.649 ms | transport + decode + GEMV |

Against N1's frozen all-cold floor of 29.608769 ms at 26.158915 GB/s, the
measured transport p50 is **100.50% of the floor time** and **99.51% of the
assumed bandwidth**. There is no transport headroom left to find and none is
needed: the 45-ms screen would pass on transport with 15.2 ms to spare.

## Transportontleding

| arm | copies issued | p50 | p95 | GB/s @p50 |
|---|---:|---:|---:|---:|
| per_record | 414 | 30.988 ms | 31.226 ms | 24.99 |
| per_layer_batched | 69 | 29.983 ms | 30.058 ms | 25.83 |
| single_contiguous | 3 | **29.756 ms** | **29.848 ms** | **26.03** |

Going from 414 copies to 3 buys **1.232 ms, 3.98%**. Dispatch overhead is
therefore about 3 µs per copy and is nearly irrelevant at this record size.

This is the direct contrast with the protected PORT80B DirectPath diagnosis,
which decomposed a 63.034 ms p50 into 37.204 ms of PCIe plus 25.830 ms of host
gather over 480 records of 2,027,520 B. Two things differ here and both were
predicted before measurement:

1. **The host gather term is gone.** The bank is built pinned once; there is no
   `mmap → pinned` bounce copy in the measured path. That term was 41% of the
   80B p50 and is 0 here.
2. **Records are 2.77× larger and 3.48× fewer** — 5,612,560 B × 138 versus
   2,027,520 B × 480 — so per-record dispatch, which mattered at 480 records,
   is now within noise.

`cudaMemcpyBatchAsync` and mapped-host/TMA paths were listed in the
preregistration as remedies to test. **They are not worth testing on this
evidence**: the remaining gap to the roofline is 0.49%, and no batching scheme
can recover a term that is already spent on the wire.

## Exactheid

| check | result |
|---|---|
| bank records bit-identical to a fresh checkpoint read | 8/8 sampled, 138/138 built without mismatch |
| GPU decode vs CPU float32 decode, `up_proj` | **bit-identical**, 4,988,928 elements, max abs diff 0 |
| GPU decode vs CPU float32 decode, `down_proj` | **bit-identical**, 4,988,928 elements, max abs diff 0 |
| expert output vs the N3-validated CPU reference | `rel_l2 = 1.859e-07` against a 1e-5 gate |
| full dequantized routed bank materialised | no |
| peak device reserved | 968,884,224 B against 8,589,934,592 |

Bit-identity is required rather than toleranced because NVFP4 decode is an
integer unpack plus two exact table lookups; both sides use the same LUTs and
the same operation order, so any difference would be a defect, not rounding.

## De echte bottleneck

`decode_only` runs the identical streaming loop with the two GEMVs removed:
353.133 ms p50, against 376.244 ms for the composed token and 29.756 ms for
transport alone.

The three numbers are deliberately **not** presented as an additive
decomposition. They do not sum: 353.133 + 29.756 = 382.889 > 376.244. The
decode-only arm forces a scalar readback per matrix to stop the decode being
elided, and that readback inserts a synchronisation the composed loop does not
have. The composed loop also interleaves per-layer transfer with compute.

Therefore the only defensible statements are:

- transport is **29.756 ms**, measured directly;
- decode is **at least an order of magnitude larger than transport**, upper-bounded at 353.133 ms;
- the residual (GEMV, allocation, loop overhead) is **not attributed**.

Naming that residual without measuring it separately is precisely the error the
project recorded after the "glue" term at ctx=128 turned out to be attention.
It is left unnamed here.

The mechanism is not mysterious. The decode expands 4,988,928 packed bytes into
9,977,856 float32 weights per matrix, through an int64 gather index, a LUT
lookup, a `repeat_interleave` of the block scales and two multiplies — roughly
120 MB of intermediate traffic per matrix, 276 times per token, to feed two
GEMVs that consume the result once. Weight-stationary arithmetic is being
performed activation-stationary.

## Meetprotocol

- Batch 1, one token, zero cache, nothing reused between tokens.
- Repetition counts fixed before results were opened: 5 warmup + 30 measured per transport arm and for the composed token; 2 + 10 for the decode-only arm.
- Wall time via `perf_counter_ns`; device time via CUDA events; both retained.
- All per-repetition raw arrays persisted, so every percentile is recomputable.
- Working set 138 records × 5,612,560 B = **774,533,280 B**, equal to N1's frozen bytes/token.
- Host bank pinned, built in the exact three-range layout measured in N2 (4,988,928 B codes + 623,616 B scales + 16 B globals per expert).
- GPU: NVIDIA RTX PRO 2000 Blackwell Laptop, CC 12.0, 8,546,484,224 B total, 7,385,120,768 B free before the run, 50 °C at start.
- torch 2.9.1+cu128, sm_120 present in `torch.cuda.get_arch_list()`.

### Non-interference

The preregistered guard initially blocked the run on "a foreign python process
exists". That rule was **wrong** and is corrected: this machine runs short-lived
python helpers with changing PIDs, and a CPU-only process contends for nothing.
The guard now blocks on the question that actually matters —
`nvidia-smi --query-compute-apps`, i.e. whether any other PID holds a CUDA
context — plus a device-memory threshold, and fails closed if the query errors.
Python process names are still recorded, as context only.

At run time: 0 foreign compute apps, 0 MiB device memory in use. No process was
killed, suspended or reniced.

## Onafhankelijke verificatie

A separate verifier re-read the raw result, recomputed every percentile from the
retained arrays, re-derived the byte accounting and effective bandwidth,
independently re-read checkpoint bytes, recomputed the CPU float32 decode from
scratch and compared its SHA-256 to the recorded GPU digest, and re-evaluated
every gate and the terminal-state logic. It imports nothing from the runner and
opens no GPU.

Result: **32/32 verification checks passed.**

Its adjudication: *the measured result is a correct negative on G4 with an
unfused decode; the architectural stop is correctly not declared, because its
preregistered precondition requires a correct fused kernel and none exists yet.*

## Gates

| # | gate | threshold | result |
|---|---|---|:--:|
| G1 | bank records bit-identical | exact | ✅ |
| G2 | GPU decode bit-identical to CPU | exact | ✅ |
| G3 | expert output vs N3 reference | `rel_l2 ≤ 1e-5` | ✅ 1.859e-07 |
| G4 | routed-expert path p95 | ≤ 45 ms | ❌ **403.649 ms** |
| G5 | no full dequantized bank | required | ✅ |
| G6 | peak device allocation | ≤ 8.0 GiB | ✅ 0.902 GiB |
| G7 | no protected byte changed | required | ✅ |

### Why the architectural stop is not declared

The preregistered stop reads: *if zero-cache expert p95 exceeds 60 ms **after
registered/batched transfer and a correct fused kernel**, reassess the physical
architecture.* Batched transfer was tested. **A fused kernel was not built** —
the decode is unfused torch ops. The precondition is unsatisfied, so the stop
must not fire regardless of how large 403.649 ms is. The runner records the
precondition explicitly and the verifier checks that it was honoured.

Declaring an architectural dead end here would have condemned the architecture
for the cost of an implementation that the preregistration itself designates as
future work.

## Eerlijk verdict

What N4 establishes:

1. The zero-cache **transport** path is solved on this hardware. 774,533,280 B
   move in 29.756 ms at 26.03 GB/s, at 99.51% of the assumed roofline, with a
   0.902 GiB device footprint and no cache of any kind.
2. Pre-pinning the bank **eliminates** the host-gather term that cost the 80B
   line 25.830 ms per token. That is a transferable mechanism, not a
   Nemotron-specific accident.
3. Transfer batching is **not** the lever here. 414 copies versus 3 differ by
   3.98%.
4. NVFP4 decode, implemented as unfused framework ops, is the wall: ≥ 10× the
   transport cost and 93.9% of the composed token.
5. The bank builder and the on-GPU decode are **bit-exact** against the
   checkpoint and against the N3-validated CPU reference.

What N4 does **not** establish: tokens per second, full-model latency, quality,
memory feasibility of the complete runtime, any comparison with another runtime,
or that a fused kernel will close the gap. A component measurement is never
promoted to tok/s — the rule that closed CORETAIL applies unchanged.

The honest reading is that H3 has produced a **useful negative with a specific,
falsifiable cause**. The physical architecture is not implicated; one
implementation is. That is the best possible shape for a failed screen, and it
converts the next phase from a search into an engineering task with a number to
beat: **the decode must come down from ~353 ms to under ~15 ms** for the routed
path to meet the 45-ms screen alongside a 29.8-ms transport.

## Vervolg

`N4-R1`, a fused NVFP4 expert kernel, before any cache work. The project already
owns the relevant prior result: ERVF/ERGV generated bit-exact kernels that
preserve a chosen reduction DAG under a changed physical topology, and the
Nemotron shapes (group 16, contraction-dim grouping, `up → ReLU² → down`) are a
clean target for the same treatment. The required property is unchanged from
this phase — bit-identical decode — and is now backed by a reference
implementation and a frozen digest to test against.

H5 cache work stays closed until the zero-cache path passes, per the assignment
and per this project's own ordering rule.

## Artefacten

- Preregistratie: `reports/lightningstream_nemotron/N4_ZERO_CACHE_DATAPLANE_PREREGISTRATION_2026-08-14.md`
- Runner: `scripts/lightningstream_nemotron/n4_zero_cache_dataplane.py`
- Machine-readable result incl. all raw arrays: `reports/lightningstream_nemotron/n4_zero_cache_dataplane.json`
- Independent verifier: `scripts/lightningstream_nemotron/n4_independent_verify.py`
- Verification output: `reports/lightningstream_nemotron/n4_independent_verification.json`
- Input lock: `reports/lightningstream_nemotron/n4_input_lock.json`
- Protected-80B check: `reports/lightningstream_nemotron/protected_verification_after_n4.json`
