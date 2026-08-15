# PORT80B-D10A2-R2 — GDN-36 CPU-oracle repair preregistration

Date: 2026-08-13

## Immutable prior and sole repair

D10A2-R stopped safely before component validation because its newly added CPU
conv oracle constructed 48 layer slices while the frozen GDN kernel and conv
buffer contain 36 layers. The immutable D10A2-R negative/blocked execution is
locked as:

- component JSON SHA-256:
  `e328d555eefa0140c6b3075d30c2ae76db4fdce0e141794ad71eab5a61ef3a7f`;
- component report SHA-256:
  `a2397b163ada4d134b47a2107d5ae7c25ff9c4abb9fb26f40ef9dbfd7617edf9`;
- runner SHA-256:
  `ea85a6e9d27627c883b6db3b63a1cdfb12040009fb89ef77be83b16eba51c275`;
- preregistration SHA-256:
  `3598c9d2da024cfe4d2ad749e7b1811982f444a3e46fd0cdcb21fa8f60da3bc0`.

That run registered 48/48 layer ranges, unregistered 48/48 successfully and
reported zero unregister failures. It is not a numerical mechanism result.

D10A2-R2 permits exactly one experimental semantic change: define
`GDN_LAYERS = 36` and use it in `conv_step0_oracle()`. The CUDA source already
uses 36 and must remain unchanged. No route, kernel, numerical threshold,
timing threshold, buffer size, resource threshold or gate may change.

## Frozen conv unit oracle

Before any GPU rerun, a CPU unit test must construct the complete step-zero BF16
conv oracle and require all of:

- flattened shape: `(1_179_648,)`, equivalently `36 × 8192 × 4` words;
- exact nonzero BF16 words: `292_608`;
- SHA-256 over the flattened little-endian `uint16` bytes:
  `cedf5736557919b023d6f7cce73d0064df07236ff1e18b5d8b3fec49d658fa1e`.

The unit-test source and its result are locked inputs to the CPU preflight.

## Inherited frozen contract

All D10A2-R requirements are inherited unchanged: one non-blocking CUDA stream
owns allocations/uploads/fills/tables/counters/kernels; same-stream header
verification; 40 correctness cases, eight warm-ups and 32 validation cases;
exact routed-Q5 and wrong-expert/layer controls; shared-payload/output oracles;
poison checks; full finite state/output digests; exactly 48 register and 48
unregister attempt rows; 52,652,163,072-byte start-RAM gate; 2 GiB post-touch
reserve; 1.5 GiB emergency floor; 4,521,569,280-byte device request plus 512
MiB reserve; p95 ≤150 ms and p99 ≤200 ms. Endurance is unconditionally closed.

## CPU-only preflight and evidence paths

The preflight must perform no CUDA initialization, NVRTC compilation, host
registration, large device allocation, kernel launch or bank scan.

- runner:
  `scripts/streamq5_moe/run_port80b_d10a2r2_gdn36_oracle_repair.py`;
- CPU unit test:
  `scripts/streamq5_moe/test_port80b_d10a2r2_conv_oracle.py`;
- unit result:
  `reports/streamq5_moe/port80b_d10a2r2_conv_oracle_unit.json`;
- preflight JSON:
  `reports/streamq5_moe/port80b_d10a2r2_gdn36_oracle_repair_preflight.json`;
- preflight report:
  `reports/streamq5_moe/PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_PREFLIGHT_REPORT_2026-08-13.md`;
- component JSON:
  `reports/streamq5_moe/port80b_d10a2r2_gdn36_oracle_repair.json`;
- component report:
  `reports/streamq5_moe/PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_REPORT_2026-08-13.md`.

## Claim boundary

Even a component pass is only a synthetic, shape-informed physical
component/composition result using proxy routes and uniform synthetic Q5
payloads. It is not an official checkpoint, natural router trace, quality,
production-throughput or endurance result.

