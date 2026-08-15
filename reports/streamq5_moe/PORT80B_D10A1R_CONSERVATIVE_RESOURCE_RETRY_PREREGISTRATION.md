# PORT80B-D10A1-R — conservative resource-only retry preregistration

## Decision and immutable base

This is the final conservative D10A1-R resource-only retry. An earlier
route-union-only preflight used 32,522,944,512 bytes; it is retained as
preliminary evidence but is explicitly non-authorizing. The independent budget
audit shows that delayed Windows/CUDA registration residency can approach the
registered-prefix scale, so D10A1-R uses the larger exact budget below.

The D10A1 runner and all experiment semantics remain locked to SHA-256
`ffde9c13a3d6d19e3e1132369a4eb9a2e98a4e974bbece86ec224e2931f0ecfd`.
Only its start-RAM constant and evidence paths are rebound. The prior D10A1
resource-stop JSON/report remain hash-locked and immutable.

## Exact conservative start budget

The runner independently replays the frozen routes and reconstructs both the
explicit touched-record union and the conservative registration-residency
budget:

```
23,952 records in the 48 × 499 registered prefix
+   428 explicitly touched records outside that prefix
=24,380 possible resident records
×2,027,520 bytes/record
=49,430,937,600 bytes = 46.03614807128906 GiB
+1,073,741,824 bytes host/process allowance
+2,147,483,648 bytes required post-touch reserve
=52,652,163,072 bytes = 49.03614807128906 GiB start gate
```

The following independent audits are mandatory SHA-256 inputs:

- `D10A1_FIRST_TOUCH_INDEPENDENT_SOURCE_AUDIT_2026-08-13.md`:
  `869e56574082e96ce960662dda3dd7e542cd814fb467bf5051831a6efefac081`;
- `port80b_d10a1r_resource_budget_audit.json`:
  `8a79cc68afa2e1e43373b9990b8cbadc9cf9b51ac811ddf2a142afea6922789f`.

The source audit's 29,301,719,040-byte explicit route union remains an exact
diagnostic but is not the authorizing bound. The budget audit's prefix-plus-
outside formula controls whenever the two interpretations differ.

## Unchanged gates

Every other D10A1 gate is copied unchanged: 2 GiB available RAM immediately
after registration and after eight first-touch warm-ups; 1.5 GiB emergency
floor during validation; exactly 4,521,569,280 allocated device bytes plus a
512 MiB VRAM reserve; the exact 499+13 bridge; all 24,576 radix-32 canary and
498↔499 checks; 40 correctness cases; raw route/canary arrays; bitexact routed
Q5 and differentiated wrong-layer/wrong-expert controls; attention, GDN,
shared-Q5, dense and runtime oracles; 8 warm-ups; 32 validation cases; wall
p95 ≤150 ms and p99 ≤200 ms; page reads ≤2,048/s; validation RAM loss ≤1 GiB;
finite state, no CUDA/runner error, and clean unregister of exactly 48 ranges.

Compile/preflight performs no host registration, large allocation, kernel
launch or bank scan. A component run requires a separate explicit GPU go.
Endurance remains fail-closed and requires a clean component pass, its own
first-touch calculation, acknowledgement and separate authorization.

## Evidence paths

- runner: `scripts/streamq5_moe/run_port80b_d10a1r_conservative_resource_retry.py`;
- compile JSON: `reports/streamq5_moe/port80b_d10a1r_conservative_resource_retry_compile.json`;
- compile report: `reports/streamq5_moe/PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_COMPILE_REPORT_2026-08-13.md`;
- component JSON: `reports/streamq5_moe/port80b_d10a1r_conservative_resource_retry.json`;
- component report: `reports/streamq5_moe/PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_REPORT_2026-08-13.md`.

## Claim boundary

A future pass remains a synthetic shape-informed physical component/composition
stress result on `p4d_shaped_synthetic_proxy` routes. It is not an exact
official Qwen3-Next shell, real checkpoint, natural routing, quality,
production-throughput, or endurance result.
