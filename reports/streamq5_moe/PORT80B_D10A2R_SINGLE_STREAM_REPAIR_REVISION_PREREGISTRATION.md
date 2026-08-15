# PORT80B-D10A2-R — immutable single-stream repair revision preregistration

Date: 2026-08-13

## Why a revision is required

The first D10A2 CPU preflight was written while its runner was still receiving
the already-preregistered evidence-strengthening edits. It correctly locked an
intermediate runner and is retained unchanged, but it cannot authorize the
final runner. D10A2-R is a new immutable revision with separate paths and a new
CPU-only preflight. No result or threshold was observed from a D10A2 GPU run;
none occurred.

The scientific hypothesis, frozen routes, kernels, numerical thresholds,
resource gates and claim boundary are identical to
`PORT80B_D10A2_SINGLE_STREAM_REPAIR_PREREGISTRATION.md` (SHA-256
`ef8a6c7ae79015d71cfa8d8031c635c56a65076f7ab4c176f6423b2bcbffd0fc`).
D10A1-R remains the immutable negative result (SHA-256
`c92e5dda380c8f9ed0669fc8961056bef58fabbf758946776b426fa7feb888ae`).

## Final frozen repair/evidence contract

D10A2-R must:

1. own one explicit non-blocking CUDA stream for all CuPy allocation,
   initialization, upload, clear/reset, pointer/route table creation, verifier
   buffers, kernels and events, with synchronization before host reads;
2. use a local same-stream header verifier;
3. preserve all 40 correctness cases, eight warm-ups, 32 validation cases,
   route transforms, Q5/component kernels, p95 ≤150 ms and p99 ≤200 ms gates;
4. preserve the 52,652,163,072-byte start-RAM gate, 2 GiB post-touch reserve,
   1.5 GiB emergency floor, exact 4,521,569,280-byte device request and 512 MiB
   VRAM reserve;
5. require full step-zero conv BF16 equality and exactly 292,608 nonzero words;
6. hash and compare all 48 shared numerical payloads against the independent
   resident reference, excluding only the three 64-byte headers, and retain the
   exact 98,304-element shared-output comparison;
7. poison routed/component/composed outputs and require all required locations
   written;
8. store full finite status, shape, dtype and SHA-256 for routed output, routed
   down output, shared output, attention, delta, KV state, recurrent state,
   conv state and composed state for every measured validation case;
9. store exactly 48 registration-attempt rows and exactly 48 unregister-attempt
   rows, and require every row successful for a pass;
10. remain unconditionally fail-closed for endurance. Even a component pass
    requires a new preregistration and runner before endurance.

The runner and this preregistration become immutable as soon as the D10A2-R
CPU preflight records their SHA-256 values. Any later byte change invalidates
that preflight and forbids component execution.

## CPU preflight boundary

The preflight may Python-compile source, replay route/canary inventories and
check immutable evidence. It must record `cuda_initialized=false`,
`nvrtc_compile=false`, `host_registration=false`,
`large_device_allocation=false`, `kernel_launch=false` and `bank_scan=false`.
Actual NVRTC compilation is deferred to a separately authorized component run.

## Evidence paths

- runner: `scripts/streamq5_moe/run_port80b_d10a2r_single_stream_repair_revision.py`;
- preflight JSON:
  `reports/streamq5_moe/port80b_d10a2r_single_stream_repair_revision_preflight.json`;
- preflight report:
  `reports/streamq5_moe/PORT80B_D10A2R_SINGLE_STREAM_REPAIR_REVISION_PREFLIGHT_REPORT_2026-08-13.md`;
- component JSON:
  `reports/streamq5_moe/port80b_d10a2r_single_stream_repair_revision.json`;
- component report:
  `reports/streamq5_moe/PORT80B_D10A2R_SINGLE_STREAM_REPAIR_REVISION_REPORT_2026-08-13.md`.

## Claim boundary

Even a pass is only a synthetic, shape-informed physical component/composition
result using proxy routes and uniform synthetic Q5 payloads. It is not an exact
official Qwen3-Next checkpoint, natural router trace, quality result,
production-throughput result or endurance result.

