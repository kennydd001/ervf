# PH1 Intel execution R8P2 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit before execution. No R8P2 import, preflight, payload read, compiler, OpenCL, or device call was performed.

## Verdict

**NO-GO for the frozen R8P2 command.**

The Windows venv/base dual-identity repair is correct and appears to close the deterministic R8P1 `exact_invocation` failure. The frozen package nevertheless has two material verification regressions and one preregistration contradiction. These require a fresh immutable preflight/verifier revision; no scientific, runtime, payload, numerical, or device logic needs to change.

## Correctly closed

- Exact handoff hashes match: preflight `4e533d5c09b9b584000b44fe2cfd04af398b96e40171ecaabc21ce05394d6b31`, verifier `6305b308d442dbc12c3b1d94e999e10a6b31e87c457d49a7bd3c4dd7689902f5`, preregistration `52f444527705b917c5abb3ea01631208362ce84df5efb64228381d52a0c3e5d9`, closed lock `b81280bd8d7cf5da093cf0cc9c937483e6b325f7d9eb782c7887d7689dfd994a`.
- All 22 source/evidence-chain hashes match the 44-key closed lock. The R8P1 diagnosis SHA `ef42c92407142893532daab1ea5dd7463bec7b384e796fcfe56df59dbbf7a6a7`, both prior audits, and the complete R8P1/R8 lineage are bound.
- The active venv launcher is 274,424 bytes, SHA `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`; `pyvenv.cfg` is SHA `9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe`; the real base binary is 172,912 bytes, SHA `5365b422ee178f691988eb937b7abca5f48910b148f76fcce6dbaf5585c948d0`. All are readable and match the lock.
- `identity_valid()` correctly separates `sys.executable`/`sys.prefix` from `_base_executable`/`base_prefix`; it requires exact native parse, full `sys.orig_argv`, full application argv, script/ACK/no extras, pyvenv fields, binary identities, and direct entry (`preflight`, lines 66–99). Raw quoting bytes are retained but adjudicated by `CommandLineToArgvW(raw)`, as preregistered.
- Invalid application argv returns 3 before the candidate failure/output writers (`preflight`, line 170). A correct-ACK internal failure enters the bounded create-new failure path (`preflight`, lines 146–167 and 171–198).
- Current lower-case R8 topology contains only the R8, R8P1 and R8P2 locks. All R8P2 result/manifest/commit/verifier/failure/quarantine/temp paths are absent.

## Blocking findings

### 1. The R8P2 transaction gate does not exercise R8P2 writers

R8P2 defines new candidate `atomic_create`, `verify_bundle`, `quarantine_core`, and `publish` functions at lines 115–144. Its actual gate at line 181 calls `prior.transaction_simulation()`, which exercises the frozen R8P1 helpers instead. A defect in any new R8P2 writer can therefore survive the reported `transaction_simulation=true` gate.

The R8P2 independent verifier similarly defines a new `atomic_create` at lines 122–129 but runs no current-verifier transaction mutation suite. Its `result_valid()` only uses `all(row["transaction_simulation"].values())` without an exact key set (line 112), so an empty transaction dictionary is vacuously accepted. The result-mutation suite does not test this case.

Required repair: add a TEMP simulator that calls the exact R8P2 production helpers, with injectable TEMP quarantine root, and covers clean result/manifest/commit, existing destination, stale temp, hard-link failure, post-link cleanup interruption, partial quarantine, repeat attempt, and unchanged committed bytes. The independent verifier must independently exercise its exact writer, require the exact transaction key set, reject an empty/missing/extra-key dictionary, and test repeated verifier preservation.

### 2. The inherited no-device gate and verifier-invocation mutations were dropped

R8P1 had a named `static_no_device` gate. R8P2's 16-check set at lines 184–189 omits it while the preregistration says all prior no-device gates remain unchanged. The current frozen source contains no compiler/OpenCL/device call, but the result provides only a hardcoded `no_compiler_device=True`; there is no current-source non-vacuity gate.

The independent verifier's live dual identity is directly checked at lines 80–82, but there is no negative suite for its own exact native/original/application vector, launcher/base split, trampoline, extras, or direct-entry state. Its nine result mutations at lines 114–120 exercise the stored preflight identity, not the verifier process contract.

Required repair: restore a current-source AST/callgraph no-compiler/no-OpenCL/no-device gate covering the candidate and inherited call path. Add an independently implemented verifier-invocation validator over an injectable row plus negative mutations for native/original/application vectors, venv/base identities, flags, trampoline, extras, wrong script, and non-direct entry.

### 3. Frozen claim wording contradicts the actual CPU-preparation arm

The preregistration ends with “It authorizes no payload,” while the same document retains CPU preparation and the candidate calls `prior.base.preparation_summary()` at line 180 and records `cpu_payload_read=True` at line 192. The intended boundary is evidently no model forward/compiler/OpenCL/device action, not no CPU payload read.

Required repair: replace that sentence in a fresh immutable preregistration with the exact boundary: CPU-only reading of the already frozen preparation payload is authorized by a later source GO; model forward, compiler, OpenCL and device actions remain forbidden.

## Minimum next revision

A bounded R8P3 or R8P2P revision may keep the dual-identity logic and every scientific/runtime binding unchanged. It needs only: exact current-helper transaction simulations and schemas, restored static no-device and verifier-invocation mutation gates, and corrected CPU-payload claim wording. The current clean topology permits that revision without cleanup, but this frozen R8P2 must not be executed.

