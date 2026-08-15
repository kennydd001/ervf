# TierFlow-F0 preregistration — route-edit traffic feasibility only

Date: 2026-08-12

## Claim boundary

This is a CPU-only trace oracle/simulator over the frozen real Qwen30 P4D
top-8 route captures. It does not train TierFlow, does not execute experts,
does not measure latency, and cannot establish language-model quality. A pass
means only that the two traffic targets are arithmetically compatible with a
bounded route-edit process on these traces.

## Locked inputs

- route capture: `reports/streamq5_moe/p4d_route_capture_result.json`
  (expected SHA-256
  `7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb`);
- route artifacts: `reports/runs/streamq5_moe/p4d_routes/layer_00.safetensors`
  through `layer_47.safetensors`, with hashes locked by the capture report;
- 48 layers, five domains, 1,024 tokens/domain, top-8 of 128 experts;
- expert-record size: 3,035,136 bytes from
  `reports/streamq5_moe/p4d_route_input_lock.json`.

The original P4D partitions are retained. Calibration 0:512 is not used.
Validation is 512:768. Test is 768:1024 and remains unopened until a single
edit budget is selected from validation.

## Definitions

For observed top-8 route set `O_t` and simulated persistent set `S_t`:

- one route edit is one replacement, equivalently one newly admitted expert;
- budget `r` enforces `|S_t \\ S_(t-1)| <= r`, with `r` in `{1, 2, 4}`;
- baseline new loads are `|O_t \\ O_(t-1)|`;
- TierFlow-oracle new loads are `|S_t \\ S_(t-1)|`;
- critical expert bytes equal new loads multiplied by exactly 3,035,136;
- route-set overlap is `|S_t intersection O_t| / 8`;
- router-output substitution rate is `1 - overlap`.

Every validation/test sequence is warm-started from the observed route at the
immediately preceding token. Cold-start bytes are reported separately and are
not used for the steady-state gates.

## Fixed oracle

At every token the oracle preserves all already-resident requested experts and
admits at most `r` missing requested experts. This maximizes overlap at the
current token conditional on the preceding state. Ties for admissions and
evictions use a fixed, non-causal within-partition Belady-style ordering:
earliest next use, then largest remaining frequency, then lowest expert ID.
It never reads across the validation/test boundary. The result is explicitly
an optimistic trace oracle, not a deployable router.

## Selection and gates

Validation evaluates all three budgets. The selected budget is the one that:

1. reduces aggregate steady-state critical expert bytes by at least 4x; and
2. reduces the aggregate worst-case new-expert load count by at least 8x;

and, among passing budgets, has the highest route-set overlap (tie: smaller
`r`). If no budget passes, test stays closed. Exactly the selected budget is
then evaluated once on test with the same two gates.

Overlap and substitution have no invented pass threshold: they quantify how
much router behavior a trained TierFlow model would have to change. The
original TierFlow quality gate (LM quality regression <=1%), p95 >=2x, no p99
collapse, and second-memory-hierarchy replication all remain untested here.

