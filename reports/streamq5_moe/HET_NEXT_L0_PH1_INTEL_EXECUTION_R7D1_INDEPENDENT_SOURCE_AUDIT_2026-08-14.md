# PH1 Intel execution R7D1 — independent final authorization audit

Date: 2026-08-14  
Scope: frozen static/read-only audit; no candidate import, payload read, compiler, OpenCL, or device call was executed.

## Verdict

**GO for exactly one physical R7D1 invocation with the exact ACK, followed by the frozen standalone R7D1 verifier.**

This authorizes one attempt only. A nonzero runner return, committed negative result, failure artifact, quarantine entry, stale temporary path, or verifier failure is evidence and does not authorize a retry.

## Frozen identities

| Artifact | SHA-256 | Status |
|---|---|---|
| R7D1 runner | `6a5fe4c1c411470cef5063802d2560bae1bb82db55323f8d605e81d33061a9c9` | exact handoff/open lock |
| standalone verifier | `e4335f5082cad73427a25ff579720b54134e2c6bcde8ea185597cbbba8959b43` | exact handoff/open lock |
| preregistration | `16d9a10d96f0e90e4088e612e9fc034bda0cd631501d8a6d63382c41552c5c11` | exact handoff/open lock |
| open lock | `ae03f3a7b720f00fbd34b85025979256e67c1efba057449168b2ffc71df30a32` | exact open token |
| R7D negative audit | `8f798ac7b5f4d98e195ac076f54aaf988c927c51cbb76d97ce19b46e72f0182f` | directly bound |

The new artifacts, R7D artifacts/audit, R7C2 PASS 9/9 result, R7A PASS 7/7 result, and R7P PASS 18/18 result all matched their lock values. The remainder of the transitive R7D-to-R0 chain is unchanged from the immediately preceding 43/43 hash audit and is bound by the new exact lock.

## Exact repair closure

- `R7A_VERIFICATION` is the exact missing path `het_next_l0_ph1_intel_execution_r7a_independent_verification.json`.
- `clean_now()` requires that path, all R7D1 output/failure/quarantine/verifier paths, and every R7D1 in-progress path to be absent.
- It also calls frozen `prior.clean_now()`, preserving the complete R7A and R7D output/failure/quarantine/verifier/temp absence contract.
- This combined live gate executes before chain hashing, `prior.authorize()`, path reassignment, recovery, payload access, backend construction, OpenCL, allocation, or launch.
- The exact path name and successful absence are retained in `r7d1_authorization` alongside the new lock SHA, complete observed chain, and exact ACK.
- The standalone verifier independently requires the R7A verifier path still to be absent, validates the R7D1 extension and lock, validates the inherited frozen R7D authorization, and only then invokes the hash-pinned R7A numerical verifier.

## No regression

- R7D1 delegates authorization to frozen R7D, so the exact R7C2 PASS 9/9, R7A PASS 7/7, R7P PASS 18/18, full sentinel/mutation evidence, and complete transitive hash chain remain mandatory.
- Physical computation, payload, OpenCL backend, numerical gates, resources, and output bundle remain the frozen R7A implementation.
- R7C2 outer failure/delegated-return handling is reused with only the R7D1 failure, quarantine, and revision paths substituted after authorization.
- Current R7A, R7D, and R7D1 physical/failure/quarantine/verifier paths were absent; matching in-progress count was zero.
- Claim remains limited to one real expert/input Intel correctness component.

## Exact authorized sequence

1. Invoke the frozen R7D1 runner once with ACK `PH1_INTEL_EXECUTION_R7D1_AFTER_R7A_VERIFIER_ABSENCE_AUDIT_GO`.
2. Preserve its exit code and every produced output/failure/quarantine artifact without cleanup or retry.
3. If a committed R7A output bundle exists, run the frozen standalone R7D1 verifier once, including when the runner classified the component negative.
4. Do not substitute the older R7A or R7D verifier.
