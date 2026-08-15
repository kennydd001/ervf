# PH1 Intel execution R7A — authorization-only revision

Status: execution-open in the frozen lock only. Physical execution remains
forbidden until an independent final source audit explicitly authorizes the
exact R7A authorization preflight and conditional runner command.

Exact acknowledgement token:
`PH1_INTEL_EXECUTION_R7A_AFTER_R7P_PASS18_AND_FINAL_AUDIT_GO`.

R7A changes no numerical, codec, kernel, buffer, launch, resource, control,
identity or claim semantics. It reuses the frozen R6 backend/common files and
the corrected R7 verifier logic. Runner/verifier changes are limited to the
fresh R7A namespace, bundle kinds, output paths, acknowledgement, and binding
the immutable R7P PASS result.

Authorization requires R7P result SHA-256
`e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959`
with kind `ph1_intel_execution_r7p_static_preflight`, `pass=true`, exactly
18/18 checks true, `no_payload_compiler_device=true`, an all-true 20-conjunct
baseline map, all 28 named verifier mutations rejected, both all-row sentinel
digests exact, and the deterministic write-after-loop negative PASS.

The open authorization preflight is itself no-payload/no-compiler/no-OpenCL/
no-device. It must pass under unchanged hashes and absent R7A output before the
physical runner may be considered. No retry or retuning is preregistered.
