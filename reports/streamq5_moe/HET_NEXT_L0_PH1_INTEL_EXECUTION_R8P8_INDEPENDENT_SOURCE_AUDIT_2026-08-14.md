# PH1 Intel execution R8P8 independent frozen-source audit — 2026-08-14

## Verdict

**GO for exactly one no-device R8P8 preflight**, using the exact frozen venv `-I -B` invocation and ACK below. This does not authorize a compiler, OpenCL, payload expansion, model forward, or device action.

No candidate import, preflight, payload, compiler, OpenCL, model, or device call was made during this audit.

## Frozen inputs

- Preflight SHA-256: `a8bb60b01bf7abb4d8a899ece281bb25f1e5276abe45aa781d025fcf7939daf6`
- Independent verifier SHA-256: `dc3204d9bce8d39280de03e969492232bf0c0a8e8faeb1c0138d978a87b48369`
- Preregistration SHA-256: `7f0f815dae9601b79550e176cf51fd840880d953249bde6104ce6ff1549d2e84`
- Closed lock SHA-256: `09f7f4f47e2eddc5ece540dbc5c7a4d4a257c5a18b6157f0994c243df7bdf302`
- Bound R8P7 audit SHA-256: `00de2f823af2a2c1f10dc8aa2239ccdc7de20ffca1cfe07a6bfe6bccf98e03fc`
- Bound R8P6 failure SHA-256: `03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372`

All delivered hashes match. The lock has exactly 76 keys, is `execution_open:false` / `audit_token:PENDING`, matches the current preflight/verifier/prereg and R8P7 audit/lock, and preserves all 68 non-metadata R8P7 lock values under the intended inherited names.

## Topology repair — PASS

- R8P1 through R8P6 are imported by exact full module names. There is no relative `.prior` chain in the topology construction.
- `GROUPS` uses R8P1's exact six `BASE_R8_PATHS`; six paths each for R8P1–R8P5; five paths for R8P6 excluding its permitted failure root; and six paths for R8P7.
- Independent literal reconstruction produced exactly 47 paths, 47 distinct paths, and zero present paths.
- The six current R8P8 paths are separate, distinct, and absent.
- The current lowercase family is exactly ten entries: the R8 lock, R8P1–R8P8 locks, and the single permitted R8P6 failure root.
- The R8P6 root contains exactly one file in one attempt directory. Its size, SHA, schema, stage/error, state, device/compiler flags, and false `direct_entry` are gated; its full SHA fixes all content.
- Root-level `inprogress` paths are forbidden. Extra family entries are rejected by exact family equality; extra/nested R8P6 evidence is rejected by the recursive one-file/one-directory check.
- Production `ancestor_contract()` requires exact group names, exact per-group cardinalities, 47/47 uniqueness, exact module identities, exact module filenames, and exact tuple equality.
- Drop and duplicate mutations cover every R8P1–R8P7 revision group; a wrong-depth substitution must also fail. The main path requires the exact mutation key set and all mutation results true.
- The independent verifier constructs the same topology from literal filesystem paths without importing the R8P8 candidate and separately validates stored and live topology.

## Retained contracts — PASS

- Current-module entry capture retains all four primitive facts and the derived conjunction. Main calls the local collector; static inspection finds no ancestor `.identity()` call. The exact 16 identity mutations remain gated.
- The production success publisher, transaction simulation, bounded failure writer, and failure simulation use the current R8P8 kinds and helpers.
- The verifier owns a separate atomic verification writer and topology construction; it does not call the candidate validator.
- Typed CPU-slice states, wheel RECORDs, dual venv/base identity, runtime mutations, 16-GiB starting-RAM gate, frozen CPU preparation/digest, R7D1/R8P1 evidence, and no-device/static boundaries remain hash-bound.
- Current R8P8 result, manifest, commit, independent verification, failure, quarantine, and temp targets are absent.

## Exact authorized action

From the workspace root, exactly once:

```powershell
& 'C:\Users\de_do\Documents\ChatGPT\New project\.venv\Scripts\python.exe' -I -B 'C:\Users\de_do\Documents\ChatGPT\New project\scripts\streamq5_moe\preflight_het_next_l0_ph1_intel_execution_r8p8.py' --ack PH1_INTEL_EXECUTION_R8P8_EXPLICIT_TOPOLOGY_CLOSED
```

Only if that process exits successfully with its exact committed bundle should the frozen independent verifier be considered next. This GO does not itself authorize that verifier or any later physical attempt.
