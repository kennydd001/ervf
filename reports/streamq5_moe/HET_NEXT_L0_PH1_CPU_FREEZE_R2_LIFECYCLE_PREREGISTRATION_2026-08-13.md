# HET-NEXT-L0-PH1 CPU-freeze R2 lifecycle revision

Date: 2026-08-13  
State at freeze: execution closed until runner plus authorization lock receive independent source GO.

R2 imports the immutable R1 wrapper SHA-256 `df824adc9072bfadec3a53570b25a531cf69493e89cde59e086dc185ce888987`, which in turn imports the immutable scientific freezer SHA-256 `746a879192041dee32acb1bcb9360ce9dde6775631c0a0671312660fb71437c8`. It changes no input, source range, codec, LUT, stage operation, metric, threshold, runtime policy, or resource limit.

Exact repairs:

1. A separate create-new authorization lock binds the R2 runner, this R2 preregistration, the R1 repair preregistration SHA-256 `61037a7d5b61cc02f818a82099ad20753cc05032afc9cb3e27ec701ca9a0a975`, R1 runner and immutable scientific base. Before any payload read or filesystem mutation, R2 checks lock schema, open state/token and every bound hash including its own source.
2. Only after the provenance gate, a valid committed final package returns immutable `already_complete`; an invalid final package and all stale R2 temp packages move to unique failed-attempt paths before payload access. There is no retry if quarantine fails.
3. Manifest names must be unique plain filenames without traversal. A valid package contains exactly the manifest-listed data files plus `manifest.json` and `commit.json`; no extra file is accepted. The commit binds manifest SHA, handoff SHA and base-result SHA.
4. Every file is fsynced. On Windows every directory promotion/quarantine/failure move uses native `MoveFileExW` with `MOVEFILE_WRITE_THROUGH`, checked return code, same-volume paths and no replacement. On POSIX, rename is followed by parent-directory fsync. The handoff records the selected durability method.
5. Failure handling preserves the original exception; it best-effort writes an fsynced failure record and write-through moves the attempt to a unique failed-attempt path. A secondary evidence error is attached without replacing the original traceback.
6. Both monkeypatches (`save_file` and inter-op setter) are restored in `finally`.

After independent source GO, the sole authorized action is one R2 CPU-only run with exact token `PH1_CPU_FREEZE_R2_AFTER_LIFECYCLE_GO`. No device, compiler, model or network action is authorized.
