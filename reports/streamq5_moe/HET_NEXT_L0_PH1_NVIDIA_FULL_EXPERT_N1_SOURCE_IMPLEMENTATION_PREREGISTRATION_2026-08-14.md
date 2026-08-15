# PH1 NVIDIA full-expert N1 source implementation preregistration

Date: 2026-08-14. State: source freeze only. Static preflight, compiler, capability and physical execution are closed.

This standalone source package implements the independently GO-audited N1 design: preregistration SHA-256 `953e3dba4158c8dafd78f8a072a8df48f924e12457509c88a3247b62faa1eb05`, capability/preflight design SHA-256 `2f9d129a2f299b15057b8da16191a6087215594863db9593a4643c661a20a90a`, design lock SHA-256 `ddbceeb31637060465464b6709f0aa530a1da7871c5ad6288a97d805efc41312`, and independent design audit SHA-256 `c55a962f0dae732b6f49947c055f97c936d27ae7d3f59e3dbf93a21467b9906f`.

## Frozen source roles

- `het_next_l0_ph1_nvidia_n1_common.py`: side-effect-free CPU package reader, official source-range quantizer, safe record checker, 22 controls, exact bit-level FP32 FMA/width-8 CPU oracle and resource contract.
- `het_next_l0_ph1_nvidia_n1_kernels.cu`: exactly two CUDA entrypoints, width-8 Q5 DAG and integer BF16 LUT activation.
- `het_next_l0_ph1_nvidia_n1_backend.py`: inert-on-import direct cdecl NVRTC one-program compiler and direct WinDLL Driver backend with immediate non-null ownership registration.
- `het_next_l0_ph1_nvidia_n1_transaction.py`: create-new, bounded bundle/failure, stale quarantine and no-retry lifecycle.
- `run_het_next_l0_ph1_nvidia_n1.py`: phase-separated sequential compile-only or physical runner; authorization precedes recovery, payload, compiler and Driver operations.
- `verify_het_next_l0_ph1_nvidia_n1.py`: independent verifier that imports none of the preceding candidate modules and independently rereads/requantizes/replays the scientific package.
- `preflight_het_next_l0_ph1_nvidia_n1_static.py`: no-payload/no-compiler/no-device source/ABI/mutation/transaction preflight.

## Exact closed phases and namespaces

Compile output is `reports/streamq5_moe/het_next_l0_ph1_nvidia_n1_compile`; physical output is `reports/streamq5_moe/het_next_l0_ph1_nvidia_n1_physical`; failure and quarantine namespaces are create-new and separately bounded. All are absent at source freeze. The source lock has `compile_open=false`, `capability_open=false`, `physical_open=false`; the preflight lock has `preflight_open=false`; the verifier lock has `verification_open=false`. Tokens are `PENDING_INDEPENDENT_AUDIT`. Invalid authorization performs no recovery or filesystem mutation.

The compile source implements one and only one `sm_120` NVRTC program, with the N1 ten-call positive ledger and explicit failure states. It retains source, log, PTX-from-sm120, ELF cubin and both disassemblies as exact manifest-bound files. Physical execution cannot compile and can consume only a later hash-frozen positive compile bundle.

The physical source implements the exact 14 pinned and 14 device allocations, 9 D8 memsets, 5 H2D, 4 launches, 9 D2H and 1 sync. Each non-null primary context, stream, module, pinned pointer or device pointer is registered as owned before its return code is adjudicated. Cleanup attempts every actually owned ordinary resource, records `owned_before/code/exception/owned_after`, then performs the R2 pop/restored-null/primary-release state machine. A positive has exactly 30 ordinary releases and eight context rows.

The resource schema has the exact N1 14 stages. Device memory is queried only at stages 6--12 while the retained primary context is current; stage 12 is before pop. Stages 13--14 are host-only with explicit `not_attempted` device fields and zero post-release Driver-context calls. Start host RAM is at least 16 GiB, later available RAM at least 2 GiB, retained peak working set at most 12 GiB, initial free VRAM at least 64 MiB and final pre-pop free is within 64 MiB of the preallocation baseline.

The physical verifier independently rebuilds the three canonical records and complete CPU oracle, then checks all five BF16 stage arrays, four all-one uint32 counter arrays, 22 controls, device identity, loader/ABI, pointer identities, copies, kernel arguments, releases, resources, terminal state and artifact topology. A positive or a frozen numerical device mismatch can be terminal evidence; authorization, provenance, protocol, lifecycle, cleanup and resource failures are invalid infrastructure evidence.

Only Python bytecode parsing is permitted before independent source audit. This preregistration authorizes no static preflight run, payload read, NVRTC load/compile, NVCUDA load, device enumeration, context operation, allocation or kernel launch.
