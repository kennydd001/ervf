# HET-NEXT-L0-C0 — capability and no-device preflight design

This is a design specification, not executable preflight source. No device calls are authorized.

## Phase 0: static/no-device preflight

The first implementation must provide a standalone preflight that imports no CUDA, OpenCL, Level Zero, SYCL, Torch, Safetensors or model runtime. It may read small JSON/Markdown/source files and safetensors headers only through a minimal local parser; it must not read tensor payloads.

It must verify:

1. self, runner, independent verifier, preregistration, compiler source and lock hashes;
2. all D2-R3, R5, C1-R2A, official shard, ST2 and D7 provenance listed in the preregistration;
3. shard exact size and a separately stored prior full SHA declaration; full 4 GB rehash belongs to the source-build phase, not static preflight;
4. exact D2 tensor keys/dtypes/shapes and the four frozen rank lists via safetensors header plus a tiny separately locked route extract; no prompt or test result access;
5. source AST contains no mutable split, threshold or sample-count argument and no device arm can override ranks;
6. exact sample schedule generator seed/algorithm, 10 warmups, 120 samples, balanced/reversed 12-sample blocks, validation-before-test state machine and test-data read prohibition before validation pass;
7. explicit inclusive host-wall boundaries around all submissions, waits, copies and host merge;
8. exact controls, call-ledger/counter schema, resource/thermal/cleanup gates and create-new transaction schema;
9. independent verifier imports no runner, kernel builder, codec or schedule helper;
10. output directories absent and no stale device allocations can be inferred from prior evidence;
11. a TEMP-only simulation of validation fail keeping tests sealed, validation pass opening tests, atomic success/failure commits, cleanup disposition and verifier nonzero exit on a false gate.

Static preflight outputs a hash-bound JSON and cannot authorize devices by itself.

## Phase 1: capability probe, separately authorized

Only after independent audit of Phase 0 may a capability-only process enumerate devices and compile minimal no-weight kernels. It performs no checkpoint payload read and no performance timing. It must capture:

- Intel exact platform/device/PCI identity, driver/OpenCL or Level Zero versions, host-USM capability bits, subgroup widths, queue profiling support, global/local memory and allocation limits;
- NVIDIA exact PCI/device identity, driver/runtime/NVRTC versions, compute capability, pinned/mapped host support, available VRAM and concurrent-kernel/copy capability;
- CPU/NUMA identity and the PCI/NUMA topology of both devices;
- proof both devices are distinct and simultaneously usable in one controlled process;
- compile logs and hashes for minimal sentinel kernels;
- allocation-only probes capped at 1 MiB/device, followed by exact cleanup and unchanged handle/memory counts.

Hard capability requirements: Intel host-USM access, subgroup width compatible with the frozen width-8 ERGV mapping, profiling events; NVIDIA required Q5 integer/BF16 primitives, pinned transfer and events; simultaneous queues without error. Failure closes C0 as `blocked_capability` without source quantization or workload execution.

## Phase 2: source-build preflight, separately authorized

Only after capability PASS may a CPU-only process rehash shard 1, reread the frozen D2 route/input/shared-gate tensors, quantize the union of the four rows' real expert triplets plus shared, and compare every source/codes/scales/decode hash against an independent verifier. No device is opened. Records remain temporary/anonymous; a compact signed manifest and CPU oracle arrays may be retained under the preregistered limits.

Before any component run, an independent source audit must confirm:

- exact real-weight tensor set and record byte arithmetic;
- CPU ERGV order, BF16 points and shared gate operand order;
- Intel and NVIDIA kernel source semantics against the CPU oracle;
- whole-expert dispatch and frozen device slots;
- timing boundaries and schedule;
- controls, resource/thermal stops, transaction recovery and cleanup;
- no reuse of D7's synthetic identical bank as real-weight evidence.

Only a new immutable execution lock may then open one physical validation attempt. Tests are opened by the frozen state machine only, never by editing a lock after observing validation timing.
