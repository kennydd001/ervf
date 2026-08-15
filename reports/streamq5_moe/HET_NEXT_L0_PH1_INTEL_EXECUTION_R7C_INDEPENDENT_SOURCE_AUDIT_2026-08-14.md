# PH1 Intel execution R7C — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only source audit. No preflight, payload, compiler, OpenCL, device, or verifier execution was performed.

## Verdict

**NO-GO for executing the frozen R7C static preflight.**

The standalone-verifier repair is sound, and the new outer boundary correctly captures exceptions that propagate out of the inherited executor. The remaining blocker is a production/preflight mismatch for late failures: real R7A device/serialization failures are caught internally and returned as code `3`, whereas R7C only creates outer evidence for raised exceptions. The preflight's late-stage fixtures raise from a replacement executor and therefore cannot detect this real return-code path.

## Frozen identities and clean state

Observed SHA-256 values match the handoff:

- runner: `0c787f16d81d92d11430fbad9a535c1cbcaf9540ab1b0c2cd7825c43cd05bd27`
- standalone verifier: `5a1741621fef0e6bab8a9f8b81f76ec72e9554a78cfb6b628c740413f3f45042`
- closed static preflight: `873f8bc0b2ecab4462f4102f6327eb53a6b46407cc438d14a6a41dc720d87205`
- preregistration: `ae47eeeef880b38804bf11427c669148fc8cbe3ca2a0936232b0482a99b52c51`
- closed lock: `febd334e1a8d4fda1fca0672820376d6cd08886f54764d40f6e179e4fda3765d`
- bound R7B audit: `20ccffff336cb57f855749e4cd1ee9e0901b39c8df91d07447ebea3d9fe902cd`

The physical R7A output, R7A failure/quarantine directories, R7C preflight result, R7C verification result, and R7C outer failure/quarantine directories are absent.

## Checks that pass source audit

1. **Authorization ordering remains fail-closed.** Exact ACK, exact 7/7 authorization-result hash/schema/content, exact closed/open lock contract when applicable, complete chain hashes, and inherited R7A authorization are processed before any configure/recovery, payload, backend construction, OpenCL load, or device action.

2. **Early outer exceptions are bounded and atomic.** Exceptions propagating from `physical.execute_authorized()` are caught. A canonical R7C failure is capped at 16 MiB, oversized evidence becomes a digest-bound summary, a unique temporary is atomically promoted, failed promotion is quarantined, and an already-valid physical commit wins without pollution.

3. **Stale handling is fail-closed.** Stale outer-failure temporaries are moved to quarantine and raise before the delegated executor is reached.

4. **Verifier independence is repaired.** The R7C verifier imports neither the R7B nor R7C runner. It defines its own paths, hashes, token, check names, lock schema, authorization-result schema, chain, and extension contract. It validates and rehashes the complete extension before importing the exact hash-bound frozen R7A numerical verifier.

5. **Verifier mutation fixtures are non-vacuous within their frozen set.** A valid independently constructed baseline is required, and all eight frozen mutations are rejected: token, result-hash, result-check, observed entry, lock hash, lock open state, R7P identity, and stage ordering.

## Blocking production/preflight mismatch

`outer_execute()` calls the inherited executor and only enters its R7C failure writer from `except Exception`. The actual executor is R7A `execute_authorized()`.

R7A `execute_authorized()` has two behaviors:

- early failures before its inner `try`—`psutil` import, start-RAM sampling/gate, payload construction, post-payload sampling, and pre-device resource gate—propagate and are correctly caught by R7C;
- backend/device, telemetry/output adjudication, serialization, bundle verification, commit/move, and oversized-attempt failures occur inside R7A's inner `try`, are archived under the immutable R7A failure/quarantine paths, and return integer `3` rather than raising.

R7C currently returns that `3` unchanged. It does not validate the inherited R7A failure artifact, retain its hash in an R7C summary, or create the preregistered R7C outer-failure bundle for the declared `device_execute` and `serialize_commit` stages.

The R7C preflight does not expose this discrepancy. For every declared stage, including `device_execute` and `serialize_commit`, it passes a replacement executor that raises `Injected`. That proves the generic exception catcher, not the real inherited late-failure route. It never simulates an executor that writes an inherited R7A failure and returns `3`, nor a `3` return without valid inherited evidence.

Consequently the preflight can report green while the production runner violates its frozen late-failure evidence contract.

## Required bounded revision

Freeze an R7C1 preflight/runner revision without changing authorization, numerical science, backend/common, thresholds, buffers, launches, or claim. It must choose and enforce one explicit contract:

- **Preferred:** when the inherited executor returns nonzero without a valid physical commit, require a new bounded R7A failure/quarantine artifact, independently verify its kind/disposition/size/hash, and atomically write a bounded R7C summary referencing that inherited evidence; reject a bare nonzero return with no valid evidence.
- **Alternatively:** narrow the preregistration honestly so late failures are evidenced exclusively by immutable R7A artifacts, then make R7C validate those artifacts and stop claiming an R7C outer bundle or exact R7C late-stage label.

The no-device preflight must TEMP-test: each propagating early exception; an inherited structured late failure followed by return `3`; a bare return `3`; an oversized inherited failure/quarantine case; stale handling through `outer_execute()` itself; valid positive and valid negative commits without pollution; and restoration/absence of all real paths.

The independent verifier design may remain, with the candidate-runner import prohibition retained.

## Claim boundary

No physical Intel result exists. This audit validates the authorization and verifier-independence repairs but does not establish PH1 correctness, performance, or a model-level result.
