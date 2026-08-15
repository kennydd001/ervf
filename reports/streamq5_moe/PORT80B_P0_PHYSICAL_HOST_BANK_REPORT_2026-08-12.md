# PORT80B_P0 - physical full-size host-bank report

## Verdict

**P0 fails under the frozen pass gates, while the physical-bank and one-hour
stability sub-gates pass.**  The current 64-GB host successfully created,
verified, memory-mapped and streamed the exact-final-size synthetic bank for
one uninterrupted hour without CUDA, runner or thermal failure.  It does not
meet the post-warm-up zero-page-read gate or the zero-cache H2D p95 gate.

The result is independently reconstructed from the saved raw arrays and
telemetry: 10/10 verifier checks pass without repeating GPU work.

## Physical bank

| item | result |
|---|---:|
| logical size | 49,925,652,480 bytes (46.496887 GiB) |
| NTFS allocated size | 49,925,652,480 bytes |
| sparse / compressed | false / false |
| build time | 35.206 s |
| full SHA256 | `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462` |
| full read-only hash warm-up | 43.760 s |
| sampled header/CRC/padding checks | 132 records, pass |

The bank is layer-major, expert-major and gate/up/down-major, with the frozen
Q5-compatible headers, 5-bit code payloads, BF16 scales and 4-KiB record
alignment.  Full `cudaHostRegister` was not attempted; all transfers used the
required eight pinned expert windows.

## Primary 10,000-token scenarios

All percentiles below are independently recomputed from the complete raw
per-token arrays.  The cache traces used the same deterministic synthetic
route generator; they are not real Qwen routing traces.

| scenario | hit rate | H2D mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| zero cache | 0.000% | 65.530 ms | 63.034 | **73.544** | 121.315 | 1,358.429 |
| 4K cache budget (2,420 slots) | 9.382% | 58.939 ms | 58.222 | 66.148 | 70.202 | 87.525 |
| 32K cache budget (2,072 slots) | 7.584% | 66.775 ms | 63.609 | 85.952 | 125.371 | 267.050 |

The frozen zero-cache gate is `p95 <= 45 ms`; measured p95 is 73.544 ms, or
1.634x the limit.  The synthetic cache improves mean latency at 4K but cannot
rescue the zero-cache gate.  The 32K trace exhibits larger paging/tail bursts.

## One-hour endurance and memory

The uninterrupted measured interval was 3,600.093 seconds.  After the three
primary traces, the zero-cache stability extension completed 24,571 tokens:

- H2D mean 67.921 ms;
- p50 65.898 ms;
- p95 79.903 ms;
- p99 94.821 ms;
- max 292.505 ms.

Final and peak process measurements:

| item | value |
|---|---:|
| final RSS | 49,886,064,640 bytes (46.460 GiB) |
| peak working set | 50,298,912,768 bytes (46.845 GiB) |
| final process commit | 1,175,638,016 bytes (1.095 GiB) |
| peak process commit | 6,074,916,864 bytes (5.658 GiB) |
| process-commit gate | <=58 GiB, pass |

The low private commit is expected for a read-only file mapping; the clean
mapped pages reside in the working set but do not become private committed
pages.

GPU telemetry remained available, with no driver reset or thermal shutdown.
Observed temperatures during logged milestones were approximately 64-74 C.

## Hard page-read telemetry

After the full-bank warm-up, 3,569 English-PDH samples were collected from the
system-wide `Memory/Page Reads/sec` and `Memory/Pages Input/sec` counters.
This is deliberately conservative: unrelated background reads also count.

| counter | mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Page Reads/sec | 49.320 | 0 | 93.406 | 1,468.221 | 7,759.005 |
| Pages Input/sec | 520.378 | 0 | 896.347 | 11,414.415 | 165,421.709 |

Because the frozen gate required **every** post-warm-up `Page Reads/sec`
sample to equal zero, it fails.  The p50 of zero shows paging was episodic,
not continuous; the tails nonetheless coincide with large transfer-latency
excursions and are disqualifying under the preregistration.

## Gate table

| gate | result |
|---|---|
| exact non-sparse physical bank | pass |
| full SHA256 | pass |
| three primary 10K scenarios | pass |
| >= one hour uninterrupted | pass |
| PDH available and sampled | pass |
| no post-warm-up page reads | **fail** |
| peak process commit <=58 GiB | pass |
| zero-cache H2D p95 <=45 ms | **fail** |
| no CUDA/runner error | pass |
| telemetry interval <=45 s | pass |
| no thermal/driver error | pass |

## Decision and claim boundary

Per the frozen hardware rule, a failure caused by post-warm-up page reads or
H2D p95 above 45 ms authorizes **consideration/testing of 96 GB RAM**.  It does
not prove that RAM alone will reduce the stable H2D p95 below 45 ms, because
the eight-window host-copy plus 480-record launch path may itself be the
dominant cost.  The next controlled experiment should therefore separate:

1. page-resident mmap-to-pinned CPU copy time;
2. pinned-window H2D time for one 973,209,600-byte token batch;
3. per-record dispatch/synchronization overhead;
4. the same trace after increasing physical RAM.

No CPU, NPU, GPU or SSD purchase is authorized by this result.

This report proves only the synthetic full-size host-bank experiment.  It is
not a real 80B model, real router, quality, hybrid-shell, prefill or end-to-end
tokens/s result.

## Artifacts

- preregistration: `PORT80B_P0_PHYSICAL_HOST_BANK_PREREGISTRATION.md`;
- runner: `scripts/streamq5_moe/run_port80b_p0_physical_host_bank.py`;
- manifest: `reports/runs/streamq5_moe/port80b_p0/port80b_p0_full_q5_bank_manifest.json`;
- raw result: `port80b_p0_physical_host_bank_result.json`;
- independent verifier: `scripts/streamq5_moe/verify_port80b_p0_physical_host_bank.py`;
- independent verification: `port80b_p0_independent_verification.json`.
