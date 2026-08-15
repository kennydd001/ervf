# PH1 Intel execution R7C1 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only source audit. No preflight, payload, compiler, OpenCL, device, or verifier execution was performed.

## Verdict

**NO-GO for executing the frozen R7C1 static preflight.**

The delegated-return repair is structurally correct: it snapshots failures, distinguishes committed output from nonzero returns, requires exactly one valid inherited failure for a positive adjudication, hashes retained files, writes a bounded atomic R7C1 summary, and keeps malformed/missing/multiple/oversized evidence negative. One evidence-integrity bug and one clean-state coverage gap remain.

## Frozen identities and absence

Observed SHA-256 values match the handoff:

- runner: `ca66c26323196c728361b30e94a0a5c8170889c6145c601d62a12ecc6db759ea`
- verifier: `17701c956d9b85291c54e4941f3c465ed7da84fe61410d3d53922f241bb361a0`
- closed preflight: `5da096d79f43bd45805e89d3197b0d2c2cb5da320678de2631aed376ba94e706`
- preregistration: `c4561800247d0cfa9cc6b6937cd121148e80b15b2b25f0f0bcc3af8a7ed4c964`
- closed lock: `047eaf9a8c3c676b9be86d2df5a0f60d5d0dd4e2b09345d14df27c9eeae0ea1c`
- bound R7C audit: `fc1c3d0b6eb1465e147e4a22f0ef8eaeb2095d5123079407a0996018caff5864`

The physical output, R7C1 preflight/verifier outputs, R7C1 failure/quarantine paths, and inherited R7A failure/quarantine paths are currently absent.

## Checks that pass source audit

1. **Snapshot and cardinality.** The runner snapshots resolved inherited `failure.json` paths before delegation and computes the exact new set afterward. Exactly one new path is required for `inherited_evidence_valid=true`; missing, multiple, malformed, oversize, or over-cardinality evidence remains negative.

2. **Inherited bundle evidence.** The entire inherited directory is recursively enumerated, capped at 32 files and 16 MiB, and represented by relative path, byte length, per-file SHA-256, failure SHA-256, deterministic canonical-row bundle SHA-256, total bytes, and file count. The failure JSON kind/status/error/device flag/disposition are schema-checked.

3. **Return and commit adjudication.** A valid committed positive or negative R7A bundle is authoritative and is not polluted. A zero return without a valid commit becomes negative. A nonzero return without a commit always produces a bounded R7C1 summary. Raised early exceptions remain covered by the outer boundary.

4. **Atomic/bounded lifecycle.** R7C1 summaries use unique temporary directories and atomic promotion. Stale temporaries are quarantined and abort; oversize R7C1 payloads become digest-bound bounded summaries.

5. **TEMP coverage.** The nine frozen cases exercise the actual R7C1 functions for early raise, one inherited failure plus `3`, bare `3`, multiple failures plus `3`, invalid oversized inherited evidence, stale quarantine, valid positive commit, valid negative commit, and zero without commit.

6. **Independent verifier.** It remains free of R7B/R7C/R7C1 candidate-runner imports, independently freezes the authorization chain and extension contract, and loads the exact R7A numerical verifier only after the R7C1 extension passes.

## Blocker 1 — false `device_opened` in valid inherited evidence

`inherited_evidence()` verifies that the inherited payload's `device_opened` value is a Boolean, but does not retain that value in its evidence record. `delegated_summary()` instead derives its own top-level value using:

```python
any(row.get("inherited_disposition") == "attempt_archived_create_new" for row in observations)
```

This is not equivalent to the inherited device flag. In particular, R7A's real oversized-attempt branch writes disposition `oversized_temp_quarantined_not_retained_failure_bundle` while retaining the actual `device_opened` Boolean from the opened backend attempt. That value is normally `true`; R7C1 necessarily records `false` for this disposition.

Thus a valid, hash-bound inherited failure can be adjudicated as valid while the R7C1 summary falsifies whether the device was opened. This violates the preregistered failure-evidence contract.

The TEMP suite does not catch it: `structured3` uses `atomic_create_new_failure_only` with `device_opened=false`, while `oversized3` creates an invalid >16 MiB failure rather than a valid small R7A oversize-summary failure with `device_opened=true`.

Required repair: retain `inherited_device_opened=payload["device_opened"]` in every parsed observation and derive the R7C1 summary directly from the validated inherited value(s), never from disposition. Add a valid late/oversize-summary fixture with `device_opened=true` and require the outer summary to preserve it exactly.

## Blocker 2 — preflight clean-state gate omits failure/quarantine paths

The frozen handoff and preregistration assert that physical output, inherited failure/quarantine, R7C1 failure/quarantine, preflight output, and verifier output are absent. The preflight's `outputs_absent` gate checks only the physical output, its own result, and the verifier result. It does not check real R7A failure/quarantine paths, R7C1 failure/quarantine paths, or matching real in-progress temporaries.

A prior/parallel failed attempt could therefore leave material evidence while this preflight still reports a clean PASS, undermining the one-attempt provenance boundary.

Required repair: the exact real-state gate must require absence of physical output, R7A failure/quarantine, R7C1 failure/quarantine, R7C1 verifier/preflight outputs, and all matching `.inprogress` paths before the no-device preflight writes its result.

## Recommended R7C2 fixtures

Retain the existing nine cases and add independently checked evidence cases:

- valid ordinary device failure with `device_opened=true`;
- valid small R7A oversize-summary disposition with `device_opened=true`;
- extra retained file whose byte length/SHA and canonical bundle digest are independently recomputed from the R7C1 summary;
- mutated inherited device flag, per-file hash, bundle hash, failure hash, and file count rejection;
- exact real clean-state absence gate.

No authorization, payload, backend/common, numerical arithmetic, stage hash, thresholds, buffers, launches, or claim change is needed.

## Claim boundary

No physical Intel result exists. This audit validates the core delegated-return mechanism but does not establish PH1 correctness, performance, or a model-level result.
