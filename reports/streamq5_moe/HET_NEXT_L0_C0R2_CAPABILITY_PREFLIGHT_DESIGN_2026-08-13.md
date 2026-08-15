# HET-NEXT-L0-C0-R2 — capability and preflight design

This design incorporates C0-R1 capability design SHA-256 `dc22fb5b65dcfd16d4899e2f0ecb00f6100d155d717bd48bafbb6167df859e92` and applies the C0-R2 specifications. It authorizes no implementation, executable preflight, device call or payload read.

The later static no-runtime/no-payload Phase-0 preflight must additionally verify:

1. exact prior C0 audit SHA `d2d33e0131b56fee2432c6945226998058495ec06bc44639bf42cba1d9767fed` and C0-R1 document hashes;
2. exact Python/PyTorch/build and five activation-runtime file hashes; source AST uses the retained CPU activation arrays as device targets and contains no native-device transcendental equivalence claim;
3. exact T0/T1/T2 schedule, explicit reverse maps and canonical pair IDs; its TEMP unit enumerates 360 observations, 120/arm, every reverse pair and the frozen FP64 quantiles/formulas;
4. exact SM64 constants/wrap semantics, byte initialization, `global_line_visit=0`, never-reset recurrence, row/phase tokens, observation indices, line traversal/mutation and digest schema; a small TEMP-array unit must match an independently frozen expected vector without allocating 256 MiB;
5. CPU-cache-only wording and absence of device-cache/VRAM eviction claims;
6. processor IDs 0/2/4, three-distinct-physical-core capability predicate, persistent two-worker topology, queue ownership, ready/start/done epoch state machine, Intel-then-NVIDIA release, arm-specific waits, fixed B `WaitForMultipleObjects` order and inclusive `t0/t1` placement;
7. p0-only guarded payload seal and exact PDH/clock gates inherited from C0-R1;
8. an independent verifier that shares none of the schedule, quantile, SM64, ledger or worker-state helpers.

The later separately authorized capability probe must report Windows processor groups/core relationships for 0/2/4 and fail if the frozen mapping is unavailable. Its <=1 MiB/device sentinel must also prove each persistent submission thread owns exactly one in-order queue/stream and no other thread submits. It still reads no checkpoint or D2 tensor payload and performs no benchmark.

The separately authorized p0 source-build must bind the exact CPU activation runtime, retain all named FP32/BF16 word arrays and use only p0-authorized byte ranges. The validation implementation audit must compare Intel/NVIDIA results to those retained CPU arrays, audit the exact three-thread state machine, and confirm the 256 MiB perturbation remains outside inclusive timing and is described only as CPU-cache perturbation.

No executable source may be written until a new independent audit returns GO on both C0-R2 documents.
