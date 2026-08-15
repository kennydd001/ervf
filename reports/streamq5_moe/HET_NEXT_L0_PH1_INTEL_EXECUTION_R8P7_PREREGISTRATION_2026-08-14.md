# PH1 Intel execution R8P7 — local-entry provenance repair

Date: 2026-08-14  
Status: immutable closed preregistration; no execution authorization.

R8P7 supersedes only the R8P6 direct-entry collector defect diagnosed by independent report SHA-256 `85d59b75a4940dd01df15d5072a0c9a1f4e9faf62260c6f8df07ed6fbfc0cba5`. R8P6 remains an immutable protocol-negative. Its sole canonical failure is SHA-256 `03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372` and its source audit is SHA-256 `bf5d8562cab9041d8a52cc9a583b0f5928617a493e27461c09c1e5af6e29d9f9`.

The R8P7 preflight computes identity locally in the current module. It retains the raw/parsed/native/original/application command vectors, venv launcher and base interpreter identities, plus the primitive entry facts `entry_name`, `entry_spec_is_none`, `entry_package`, and resolved `entry_file`. `direct_entry` is exactly the conjunction `entry_name == '__main__'`, `entry_spec_is_none is True`, `entry_package in {None, ''}`, and `entry_file == resolved R8P7 SCRIPT`. The current main path must call this local collector. Static AST inspection rejects any ancestor `identity()` call. Wrong name, non-null-spec evidence, nonempty package, wrong file, false conjunction, wrong script/ACK, extra arguments, reordered flags, and trampoline forms are all negative fixtures.

The pre-run topology permits exactly the single frozen R8P6 failure file in its one attempt directory and independently validates its SHA, size, schema, error, state, no-device fields, and sole false `direct_entry` field. Every R8P6 result/manifest/commit/verifier/quarantine/temp and every R8P7 result/failure/quarantine/temp remains absent. Missing, extra, altered, oversized, or malformed R8P6 failure evidence fails closed.

All consolidated R8P6 transaction, typed CPU-slice state, bounded failure, runtime, wheel RECORD, 16-GiB RAM, frozen CPU preparation, 22-control, five-stage-hash, R7D1/R8P1 provenance, static no-device and claim-boundary gates remain unchanged. The current R8P7 success writer and its TEMP transaction suite use the same R8P7 helpers; the independent verifier owns its writer/TX implementation.

R8P7 is closed/PENDING. No model forward, compiler, OpenCL, CUDA, payload expansion, physical backend, or device action is authorized. A later independent source GO may authorize one exact no-device preflight only.
