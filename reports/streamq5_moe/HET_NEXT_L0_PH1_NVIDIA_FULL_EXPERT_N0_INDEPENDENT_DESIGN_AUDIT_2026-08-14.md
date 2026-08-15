# PH1 NVIDIA full-expert N0 — independent design audit

Date: 2026-08-14  
Scope: read-only audit of the frozen N0 preregistration, capability/preflight design, and design lock. No implementation, import, payload read, compiler, CUDA library load, preflight, or device call was performed.

## Verdict

**NO-GO for implementation from the exact N0 design.** The scientific input, arithmetic, buffer schedule, result gates, lifecycle intent, controls, resources, and narrow claim are strong. Four bounded contract defects remain: an impossible post-context device-memory query, an ambiguous PTX/cubin compile protocol, missing exact Driver-call operands, and an incomplete direct PH1-R1 provenance lock.

These are pre-implementation design defects, not mechanism negatives. No NVIDIA scientific conclusion exists yet.

## Frozen design package

- preregistration: `9d86bab43d1a2c16f90cdf5d9d67f073f5dd887c2ba44baaae0452b03f43175c`, 13,776 bytes;
- capability/preflight design: `b01fc7818f74a45b00924f21b0b991a529bebd7ba5f95dafdb96769d75736efc`, 8,890 bytes;
- closed design lock: `9965a1bbaa0a591a479ef4fba7f9d349da34f1315fa9d7bebd37854d397dd778`, 1,599 bytes.

All lock hashes supplied by the handoff match their current files. The design lock correctly has implementation, compiler, and device phases closed. Before this audit file was written, the N0 family contained only the three preregistered design files.

## Correctly frozen scientific contract

The design binds the terminal Intel verifier SHA `42cd69582a47b8b5f8f4b7f24a696f1d3fcc6fbd49c05d0f61354a57cefc052d` and exact R8A5 result/manifest/commit hashes. It uses the same official expert-50 records, D2R3 `p0_n16` input, normative LUT, CPU package and five stage hashes as the formal Intel component.

The three records are exactly 675,840 bytes each with shapes gate/up `[512,2048]` and down `[2048,512]`. Input is 4,096 bytes; LUT is 131,072 bytes. The Q5 field/scale/header/CRC contract and CPU source-versus-Q5 quality boundary are unchanged.

The proposed width-8 arithmetic matches the frozen CPU oracle:

- each pack evaluates eight fields in increasing order with explicit round-to-BF16 dequantization and FP32 FMA;
- per-lane pack partials use the frozen `16,8,4,2,1` or `4,2,1` reduction tree;
- the eight lanes reduce `4,2,1` with round-nearest FP32 adds;
- lane zero rounds once to BF16;
- activation uses raw-word LUT lookup plus the frozen exact integer BF16 multiply;
- all five stage arrays and all four counter arrays must equal CPU and immutable Intel evidence byte for byte.

The host/device buffer table is internally consistent: 14 buffers, 2,185,216 bytes per memory class and 4,370,432 bytes combined requested pinned/device memory. The schedule is exact and shape-consistent: 9 memsets, 5 H2D copies, 4 launches, 9 D2H copies and 1 stream sync. Launch geometries cover exactly 512 gate/up/activation rows and 2,048 down rows.

The design also correctly freezes 22 predevice controls, exact device identity/PCI separation, 30 ordinary releases, eight separate primary-context rows, 14 host-resource samples, raw evidence, bounded artifacts, failure classification, no retry/retune/fallback, and no concurrency or performance claim.

## Blocking findings

### 1. `cuMemGetInfo` is required after the context is no longer current

The preregistration line 101 says device-memory information is retained “from post-push through post-context-release.” Yet lines 87–89 adopt the R2 lifecycle: pop the owned context, prove restored current context is `NULL`, release the primary context last, and forbid any use afterward. The frozen R2 contract line 24 explicitly rejects any context use after release.

`cuMemGetInfo` reports memory for the **current context** and may return `CUDA_ERROR_INVALID_CONTEXT` when no valid context is current. The CUDA Programming Guide likewise states that context-operating Driver calls require a current context. See the official [Driver API memory documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/group__CUDA__MEM.html) and [Driver API programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html).

Required repair: freeze exact device-memory sample stages. The final normative sample must occur `post_ordinary_releases_pre_pop` while `owned` is still current. After pop/restored-null/primary-release, the host resource sample must record device-memory fields as `not_attempted`/null and prove no Driver context operation occurred. Compare the final pre-pop free value with the preallocation value for the 64 MiB diagnostic.

### 2. The compile protocol mixes `compute_120` and `sm_120`

Preregistration line 93 freezes one option vector containing only `--gpu-architecture=sm_120`, but simultaneously calls the textual artifact a diagnostic `compute_120` PTX. Design line 29 retains PTX and cubin from an unspecified program cardinality. This leaves source implementation free to choose one or two NVRTC programs, option vectors, logs and destroy paths.

NVRTC permits PTX retrieval and cubin retrieval after compilation, while cubin is nonempty only for a real architecture such as `sm_120`; virtual `compute_120` produces no cubin. See the official [NVRTC 13.3 documentation](https://docs.nvidia.com/cuda/nvrtc/).

Required repair: choose and freeze one of these protocols:

1. one `sm_120` program, one compile, one log, both `nvrtcGetPTX*` and `nvrtcGetCUBIN*`, one destroy; call the textual output “PTX retrieved from the `sm_120` compile,” not a `compute_120` compile; or
2. two programs with exact `compute_120` and `sm_120` option vectors, two logs and two destroy attempts, with explicit artifact identities and failure behavior.

The preflight/compiler verifier must hardcode the selected call cardinality and reject the other.

### 3. Several scientific Driver-call operands are not frozen

The design freezes API names and high-level row counts but not all operands that materially determine behavior:

- `cuMemHostAlloc` flags are unspecified despite mapped zero-copy being forbidden;
- `cuStreamCreate` flags are unspecified despite default-stream interaction being forbidden;
- `cuModuleLoadDataEx` option count/options/value pointers are unspecified for the precompiled cubin;
- `cuLaunchKernel` shared-memory bytes and `extra` pointer are unspecified;
- the NVCUDA loader path/hash and Windows loader convention are not yet normative;
- “runtime version drift” is a gate even though CUDA Runtime/CUDART loading is forbidden and therefore no runtime version is available.

Required repair: freeze `cuMemHostAlloc(..., flags=0)`, one exact nondefault-stream flag policy, `cuModuleLoadDataEx(..., numOptions=0, options=NULL, optionValues=NULL)`, `cuLaunchKernel(..., sharedMemBytes=0, kernelParams=<stable array>, extra=NULL, stream=<owned nondefault>)`, exact NVCUDA DLL identity/loader ABI, and driver/NVRTC/tool versions only. Record `cudart_loaded=false`; remove the nonexistent runtime-version gate.

### 4. The lock omits the directly inherited PH1-R1 contract and audit

The N0 lock directly binds the PH1-R2 context addendum/audit, but not the PH1-R1 physical contract SHA `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981` or its audit SHA `cb295f83e5a49ebdccce9982342af4442fa4a1fbb3607d547a5a6804eaa97cfe`. R2 expressly supersedes only the context-lifecycle ambiguity and inherits all other R1 arithmetic/buffer/copy/launch/control/claim terms.

Required repair: add direct PH1-R1 contract and audit hash fields to the revised design and every downstream source/preflight/compile/physical lock. Retain the R2 hashes as the narrow lifecycle supersession.

## Required N0-R1 design revision

Freeze a bounded N0-R1 design-only revision that changes only:

1. device-memory sampling before pop/release;
2. exact one-program or two-program NVRTC protocol;
3. complete Driver operand/loader/version constants;
4. direct PH1-R1 contract/audit bindings plus this audit.

All scientific input hashes, Q5/BF16 arithmetic, stage hashes, buffer/copy/launch schedule, counters, controls, primary-context ownership, cleanup, resource bounds, terminal classification and narrow claim must remain unchanged. The revision requires another independent design audit before implementation.

## Claim boundary

No NVIDIA mechanism result exists from N0 design alone. A future positive may claim only exact reproduction of one official real expert-50/input by the pinned NVIDIA direct-cubin path. It cannot claim performance, concurrency, heterogeneous cohabitation, router/layer/model correctness, deployment readiness, novelty or breakthrough.

