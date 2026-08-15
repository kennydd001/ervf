# HET-NEXT-L0-C0-R3 — capability/preflight design

Design only; no executable source, runtime import, tensor payload read or device action is authorized.

The future static Phase-0 preflight incorporates C0-R1/R2 requirements and must additionally verify:

1. exact C0-R1/R2 document hashes and original audit SHA;
2. normative BF16 input/output contracts for direct `F.silu(BF16)` and `torch.sigmoid(BF16)`, including a CPU-only TEMP unit that asserts BF16 returned dtype and frozen raw-word fixtures; any FP32 diagnostic is named/non-normative and has no dataflow into gates;
3. source AST/dataflow requires each device's own gate/up output through its own activation into its own down operation inside every timed sample; CPU activation arrays can only enter verifier comparisons and cannot be device inputs;
4. logical processors 0/2/4/6 on distinct physical cores; exact LP roles and exclusive queue/stream/PDH ownership;
5. only Win32 `CreateEventW` primitives with frozen auto/manual reset types, SRWLOCK descriptor publication, epoch/ack rules, reset ordering, 30-second timeouts, A/S/B command-ready-start-done sequences and exact QPC positions;
6. a separate LP6 waitable-timer/PDH lifecycle, 100 ms deadlines, `[80,120] ms` interval and 20 ms lateness validity rules, exact 2-second row windows and handle cleanup;
7. TEMP-only pure state-machine simulation of all A/S/B event transitions, a stale epoch, a timeout, distinct-worker release, monitor deadline boundaries and verifier rejection of one false transition—without importing or calling any device/runtime API;
8. independent verifier shares no event-state, activation, schedule, quantile, thrash, PDH or transaction helper with the runner.

The later separately authorized capability probe may only enumerate the frozen CPU topology/devices and compile/allocate the prior <=1 MiB sentinels. It must prove the two worker queues and LP6 monitor can coexist and clean up without reading weights or benchmarking. The later p0-only CPU source phase retains normative BF16 arrays. A later source audit must inspect each device activation/down dataflow and all Win32/QPC/PDH paths before one physical validation can be authorized.
