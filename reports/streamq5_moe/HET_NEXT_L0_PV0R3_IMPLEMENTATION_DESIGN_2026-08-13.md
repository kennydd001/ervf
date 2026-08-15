# HET-NEXT-L0-PV0-R3 — minimal implementation design revision

## Frozen design bundle

This non-executable design binds:

- R3 preregistration
  `reports/streamq5_moe/HET_NEXT_L0_PV0R3_REAL_WEIGHT_PROCESS_VALIDATION_PREREGISTRATION_2026-08-13.md`;
- unchanged 33-record manifest, 22,287 bytes, SHA-256
  `0e8882943590e5bb5c9a9d26bdb89e90963c6f732e707bae78f6f50c18cfee40`;
- R2 prereg/design hashes
  `66c243b2b0ec52ff1f9cea385c669a0f7256f7bc76e0fee1d829a1d0c45c0fe6`
  and `5fa36c2b19c161f86386ad0f2fa9e815bb9376a18b4d74da2f34986392033cca`.

A future runner lock must bind the final byte hashes of this R3 preregistration,
this design and the manifest. No implementation or output exists now.

## Source stage

The builder and independent verifier implement separate selected-source graph
code. Their full-16 selected-subgraph arrays must match bitwise. The only routed
D2 comparison slices exact token index 15 before comparing; any source access or
gate that compares selected routed tokens 0-14 with full D2 experts is forbidden
by source AST and result schema. Shared raw remains a complete full-16 bitwise
D2 comparison.

The 33-record manifest, D2 p0-only range reader, ties-to-even Q5 codec, 23-hit
gathers, source/Q5 intermediates and predevice control witnesses remain R2.

## NVIDIA stage

Implement the 15-row R3 allocation table as a frozen machine-readable tuple.
Require sum 28,713,088, retained-stage decomposition 90,112+131,072=221,184
and outbound decomposition 24,576+4,096+32=28,704. Capture pointer identity,
allocation/free calls and every copy direction/byte count.

The shared gate path has separate pinned-linear, device-linear and
device-sigmoid 32-byte objects. Kernel order is copy linear, device sigmoid to
BF16, device gate-first multiply with shared raw, synchronize, then copy sigmoid
and shared-gated evidence out. The shared down device buffer is produced from
the device's own shared activation. CPU arrays never enter this sequence.

## Independent adjudication

The verifier independently checks exact raw keysets, the selected/full D2
comparison boundary, all 33 source/codec records, the NVIDIA allocation table,
shared-gate pointer/call order, Intel host-USM semantics, per-stage metrics,
official-ID merge, controls, phase RSS, process lifecycle and cleanup. Stored
booleans and allocation totals are not trusted.

Only a hash-bound verifier positive can precede commit-last. A design GO opens
runner/verifier implementation only; it does not authorize payload or device
execution.
