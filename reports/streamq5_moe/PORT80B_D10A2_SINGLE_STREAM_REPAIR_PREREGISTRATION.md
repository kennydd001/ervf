# PORT80B-D10A2 — deterministic single-stream repair preregistration

Date: 2026-08-13

## Immutable prior and decision

D10A1-R remains an immutable negative with endurance closed. Its raw result is
locked to SHA-256
`c92e5dda380c8f9ed0669fc8961056bef58fabbf758946776b426fa7feb888ae`.
This new arm is justified by the independent failure counter-audit, locked to
SHA-256
`75f5ac247a2c270f5f2d0480cbd63833ad3d1755b509a7929946c948b95fc5e1`.
The D10A1-R runner and preregistration are locked respectively to
`a9bb549b6f7a21dfedaf28a44b8e249b28a8b75747502e2113d31fe02f5c189d`
and `d606b438595a3aba7f4a1fb11aa93c97bf42a1a63f544138a5477c7c94fc62c7`.

D10A2 tests one causal hypothesis: the three D10A1-R failures came from
default-stream producers racing consumers on a separately created non-blocking
stream. D10A2 may repair ordering and strengthen evidence, but may not tune
routes, thresholds, numerical kernels, resource budgets or timing gates.

## Frozen mutation whitelist

The runner may only:

1. place every device allocation/initialization, `cp.asarray` upload, pointer
   table, route array, counter clear, state reset and kernel launch in the same
   explicit `cp.cuda.Stream(non_blocking=True)` context;
2. synchronize that stream before every host read;
3. replace the imported cross-stream header verifier with a local same-stream
   verifier;
4. strengthen the step-zero conv oracle from `nonzero > 0` to an exact full
   BF16-word comparison, including exactly 292,608 nonzero words and SHA-256
   digests;
5. compare every shared record's numerical payload with the resident reference,
   excluding the three 64-byte headers, and retain the exact 48 × 2,048 shared
   output comparison;
6. poison required component output buffers with a frozen finite sentinel and
   require no sentinel to remain after the component kernels;
7. store full finite checks and SHA-256 digests for state and all composed
   validation outputs after each of the 32 measured cases;
8. store one registration-attempt row and one unregister-attempt row for every
   one of the 48 layer ranges.

No CUDA formula, Q5 layout, route transformation, route partition, number of
cases, warm-up count, timing boundary or threshold may change.

## Frozen execution and gates

- identical `p4d_shaped_synthetic_proxy` routes;
- correctness partition 0:8 over five domains: exactly 40 cases;
- validation partition 512:576, truncated exactly to 32 cases;
- eight first-touch warm-ups;
- 48 layers, top-k 10, 499-record registered prefix and at most 13 cold
  records/case;
- conservative start-RAM gate 52,652,163,072 bytes;
- post-registration and post-first-touch RAM at least 2 GiB;
- emergency validation RAM floor 1.5 GiB;
- exact device request 4,521,569,280 bytes plus at least 512 MiB free reserve;
- wall p95 at most 150 ms and p99 at most 200 ms;
- no page-read sample above 2,048/s and validation RAM loss at most 1 GiB;
- original canary, exact routed-Q5, differentiated negative-control, attention,
  dense/runtime, registration and clean-unregister gates all remain;
- strengthened conv, shared-payload, sentinel, full-finite and digest gates are
  additive and cannot weaken an original gate.

A component pass requires every gate true, no runner/CUDA error, exactly 48
successful registration rows and exactly 48 successful unregister rows.
Endurance remains closed even after a component pass and requires a new
authorization/preregistration.

## CPU-only preflight

`--phase compile` is deliberately a CPU-only static preflight. It may parse and
Python-compile source, replay route inventory and immutable audits, and inspect
the mutation contract. It may not initialize CUDA, compile NVRTC, allocate a
device buffer, launch a kernel, register host memory or scan the 49.9 GB bank.
CUDA module compilation is deferred to a separately authorized component run.

## Evidence paths

- runner: `scripts/streamq5_moe/run_port80b_d10a2_single_stream_repair.py`;
- CPU preflight JSON:
  `reports/streamq5_moe/port80b_d10a2_single_stream_repair_preflight.json`;
- CPU preflight report:
  `reports/streamq5_moe/PORT80B_D10A2_SINGLE_STREAM_REPAIR_PREFLIGHT_REPORT_2026-08-13.md`;
- component JSON:
  `reports/streamq5_moe/port80b_d10a2_single_stream_repair.json`;
- component report:
  `reports/streamq5_moe/PORT80B_D10A2_SINGLE_STREAM_REPAIR_REPORT_2026-08-13.md`.

## Claim boundary

Even a pass is only a synthetic, shape-informed physical component/composition
result using proxy routes and uniform synthetic Q5 payloads. It is not an exact
official Qwen3-Next checkpoint, natural router trace, quality result,
production-throughput result or endurance result.

