# PH1 Intel execution R7B — authorization-result-bound revision

Date: 2026-08-14

## Scope

R7B is an authorization-only wrapper around the immutable R7A physical execution. It does not alter the backend, common package, kernels, buffers, launch geometry, controls, numerical gates, resources, output bundle, or claim. The physical bundle remains `het_next_l0_ph1_intel_execution_r7a`; the post-run verifier is the R7B authorization-chain verifier, which reuses the frozen independent R7A numerical verifier and independently adjudicates the R7B authorization extension.

## Mandatory pre-execution gate

Before any call to R7A `configure`, recovery, payload construction, backend construction, OpenCL load, allocation, or launch, R7B must:

1. require its exact ACK `PH1_INTEL_EXECUTION_R7B_AFTER_R7A_PASS7_AND_AUTH_AUDIT_GO`;
2. hash the immutable R7A authorization-preflight result as `a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265`;
3. require the exact result kind, exact eight top-level fields, seven exact check names all true, `pass=true`, `passed=total=7`, `no_payload_compiler_device=true`, the exact R7A ACK, and R7P result hash `e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959`;
4. rehash and match every R7B lock binding, including the immutable R7A runner/verifier/preflight/prereg/lock, R7A audit, R7P chain, R7 corrected runner/verifier, R6 backend/common, and R0 backend/runner;
5. replay the complete immutable R7A transitive authorization before delegating to `execute_authorized`.

Any mismatch returns nonzero without filesystem mutation. The R7B authorization evidence is retained inside the physical result alongside the inherited R7A authorization and must be independently checked after the run.

## Immutable inherited science

R7B changes no scientific or device behavior. The claim remains: one real expert/input Intel correctness component only. It proves no performance, model-level quality, heterogeneous co-execution, or production deployment property.

## Execution state

The R7A authorization preflight is immutable PASS 7/7 and device-free. R7B physical execution remains closed until an independent source/hash audit of this exact revision explicitly authorizes one attempt.
