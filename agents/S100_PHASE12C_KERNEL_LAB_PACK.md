# S100 Phase 12C kernel lab pack

Parent evidence: `1280bc6`.

Phase 12A proves exact B=2/4/8 block semantics but ordinary M=1 kernels cost 35.6/71.0/143.9 ms. Phase 12B opens grouped MoE with 29.2% median routed device-read reduction at B=4 and 43.8% at B=8.

The external one-click pack tests the two remaining hardware gates before integrated verifier work:

1. shared-weight exact ERVF-M over all real resident BF16/FP8/NVFP4 matrices;
2. real-route grouped expert-up plus actual-activation grouped sparse-down.

Frozen gates:

- dense ERVF-M: 1.75x / 3.20x / 5.50x at B=2/4/8;
- grouped up: 1.20x at B=4, 1.35x at B=8;
- grouped down: 1.15x at B=4, 1.25x at B=8;
- every compared result bit-identical.

`PHASE12C_INTEGRATION_OPEN` requires all B=4 gates and complete instrumentation. A component result is never an S100 claim.

Pack: `ervf_s100_phase12c_kernel_lab.zip`

SHA256: `12015b045632092a8a3858acb029e3f7cd273fb1b99bde9cc2e62060f4642686`
