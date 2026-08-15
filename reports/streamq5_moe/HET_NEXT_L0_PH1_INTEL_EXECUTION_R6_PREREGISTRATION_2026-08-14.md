# PH1 Intel execution R6 — two-point coverage erratum

Status: closed/PENDING. No preflight, payload, compiler, OpenCL load, or device call is authorized.

R6 binds R5 independent audit SHA-256 `df59ed6dfbdba95517243316aa70780ab52bf2c3a103946b06c4e0a258fb20e4`. It makes exactly two lifecycle/test-coverage repairs and no codec, kernel, buffer, launch, threshold, device-identity, arithmetic, or claim change.

1. For host-USM type, base, and size attestation failures, the TEMP fixture follows production order: a non-null allocation is first appended to `self.allocations` under `usm:attest_<field>`, then the attestation failure is injected. Cleanup must issue that promoted reverse free exactly once, must not also issue a `pending_usm` free, and must end with zero live resources. The allocation-status failure remains pending, matching production where status is checked before promotion.
2. The runner and independent verifier require exactly 42 `clGetMemAllocInfoINTEL` and 18 `clSetKernelArgMemPointerINTEL` ownership rows and require every return code to equal integer zero. The production-verifier fixture mutates one getInfo and one set-pointer code to nonzero and each mutation must make verification fail.

All R5 failure, resource, transaction, ownership-crosslink, bundle, provenance, control, and numerical checks remain unchanged. The 16 MiB cap still applies to `OUT` and `FAILED`; `QUAR` remains explicitly forensic evidence outside the scientific/failure bundle.
