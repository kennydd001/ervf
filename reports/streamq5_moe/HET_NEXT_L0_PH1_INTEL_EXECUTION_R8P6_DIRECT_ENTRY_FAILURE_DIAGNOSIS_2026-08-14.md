# PH1 Intel execution R8P6 direct-entry failure diagnosis — 2026-08-14

## Verdict

R8P6 is an immutable **valid protocol negative** caused by a deterministic caller-context bug in the preflight harness. It is not a runtime, CPU-evidence, compiler, OpenCL, or device failure. No retry or reclassification is justified.

The sole failing invocation predicate is `direct_entry`. The intended command was in fact used, but R8P6 obtains identity evidence by calling an identity function defined in an imported R8P4 module. That function evaluates `__spec__` and `__package__` in R8P4's defining module namespace. Since R8P4 is imported, its `__spec__` is non-null, so it deterministically reports `direct_entry: false` even while R8P6 itself is the directly executed `__main__` module.

## Frozen evidence

- Failure: `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8p6_failed_attempts/attempt_71e198678f004a56a6912d07a4187dfd/failure.json`
- Failure SHA-256: `03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372`
- Failure bytes: `2,986`
- R8P6 preflight SHA-256: `f67030aaf9cfeac5266bb3a59971b8589692a6b96781413ed56b599a9975c6e5`
- Failure schema: exactly 12 keys; identity schema: exactly 13 keys.
- Status/stage/disposition: `valid_protocol_negative` / `identity` / `bounded_create_new_canonical`.
- Error: `RuntimeError:exact_invocation`.

The canonical raw command is byte-for-byte the expected command. `native_argv`, `sys.orig_argv`, and `sys.argv` equal the frozen vectors. The venv launcher, venv prefix, base interpreter alias, base prefix, launcher hash, `pyvenv.cfg` hash/content, base-binary hash, and base-binary byte count all match. An independent field comparison found every invocation subcheck true except `direct_entry` (11/12 boolean comparisons, with `native_raw` separately exact).

## Exact source cause

- R8P6 line 173 calls `prior.prior.identity()`.
- `prior` is R8P5, so `prior.prior` is the imported R8P4 module.
- R8P4 line 39 computes `direct_entry` as `__spec__ is None and (__package__ is None or __package__ == "")`.
- Those globals belong to imported R8P4, not directly executed R8P6. Therefore the result is necessarily false.
- R8P6 line 174 requires the retained value to be true and raises before topology, runtime collection, frozen CPU-slice preparation, compiler, OpenCL, or device work.

This is not evidence that the shell, venv launcher, `-I -B`, script path, ACK, or Python trampoline was wrong.

## Zero-work and topology boundary

The failure retains:

- `cpu_slice_state: not_started`
- `cpu_frozen_slice_read_started: false`
- `cpu_frozen_slice_read_completed: false`
- `compiler_opened: false`
- `device_opened: false`

No R8P6 result, manifest, commit, independent-verification file, quarantine, or `.inprogress` file exists. The R8P6 family contains only the frozen lock/prereg/audit material and this single bounded failure bundle.

## Minimal R8P7 repair contract

R8P7 may be a provenance-only successor, but it must not erase or reinterpret R8P6.

1. Capture direct-entry evidence in the **current R8P7 module**, before topology, CPU-slice reads, compiler, OpenCL, or device access. Do not call any ancestor's `identity()` for module-entry state.
2. Retain the primitive current-module facts, not only a derived boolean: `__name__ == "__main__"`, `__spec__ is None`, `__package__ in {None, ""}`, and resolved `__file__ == SCRIPT`. Derive `direct_entry` only as their conjunction.
3. The OS/process portion may be shared only as a context-free parser/capture helper. Current-module entry facts must be supplied by or evaluated in R8P7 itself.
4. R8P7's validator must require the exact raw/parsed/orig/sys argv vectors, both venv/base identities and hashes, all retained local-entry primitives, and `direct_entry is True`.
5. The current R8P7 main path itself is the nonvacuous real-main positive: it must evaluate the local facts under the exact authorized `-I -B <R8P7 script> --ack <R8P7 ACK>` invocation. Negative fixtures must independently reject wrong `__name__`, non-null `__spec__`, nonempty package, wrong `__file__`, wrong script/ACK, extra argv, reordered flags, and trampoline forms.
6. Static/call-graph checks must prove that R8P7 defines and calls the local entry capture and contains no `prior.*identity()` call for `direct_entry`. A fixture that merely supplies `direct_entry=True` is insufficient.
7. The independent verifier must reconstruct the exact vectors/hashes and inspect the frozen R8P7 source independently; it must not import or trust the candidate identity validator.
8. The R8P7 topology must explicitly allow and hash-bind exactly the one immutable R8P6 failure file above, while requiring all new R8P7 result/failure/quarantine/temp targets absent before authorization. Multiple, missing, altered, oversized, or malformed R8P6 failure bundles must fail closed.
9. Preserve the R8P6 CPU-state, transaction, failure-writer, runtime, wheel, and science contracts unchanged. The repair authorizes only a fresh R8P7 no-device preflight after a new frozen source audit; it does not authorize a physical OpenCL attempt.

## Claim boundary

Safe conclusion: R8P6 never reached frozen CPU evidence or any device/compiler operation because of an imported-module direct-entry evaluator bug. An R8P7 repair can close that preflight-provenance defect. It cannot convert R8P6 to a pass, establish Intel execution eligibility, or support any numerical/performance/model claim by itself.
