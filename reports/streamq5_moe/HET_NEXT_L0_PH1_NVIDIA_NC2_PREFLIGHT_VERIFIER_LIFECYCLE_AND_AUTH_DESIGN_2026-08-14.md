# NC2 preflight, verifier, lifecycle and authorization design

## Runtime and phase-local authorization

Every future executable is invoked through `C:\Users\de_do\Documents\ChatGPT\New project\.venv\Scripts\python.exe -I -B`. The launcher is 274,424 bytes/SHA-256 `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`; `.venv\pyvenv.cfg` is 477 bytes/SHA `9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe`; Python is CPython 3.12.10. `sys.prefix` is the workspace `.venv`; `sys.executable` is that launcher. `sys._base_executable`, native argv[0] and `sys.orig_argv[0]` are the exact WindowsApps alias `C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe`; `sys.base_prefix` is `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0`; its real `python.exe` is 172,912 bytes/SHA `5365b422ee178f691988eb937b7abca5f48910b148f76fcce6dbaf5585c948d0`.

Raw `GetCommandLineW` is retained. Only `CommandLineToArgvW(raw)` equality to the exact native vector is normative; raw quote bytes are not. `sys.orig_argv` equals that vector, `sys.argv` equals the script-and-arguments suffix, and direct entry requires `__name__="__main__"`, `__spec__ is None`, `__package__ in (None,"")`, and resolved `__file__` equal to the current script. Imported entry, `-c`, `-m`, trampoline or extra argument fails mutation-free.

Frozen future ACKs are:

- static preflight: `ACK_HET_NEXT_L0_PH1_NVIDIA_NC2_STATIC_PREFLIGHT_AFTER_SOURCE_AUDIT_GO`;
- compiler: `ACK_HET_NEXT_L0_PH1_NVIDIA_NC2_COMPILE_ONLY_AFTER_PREFLIGHT_AND_AUDIT_GO`;
- postcommit verifier: `ACK_HET_NEXT_L0_PH1_NVIDIA_NC2_POSTCOMMIT_VERIFY_ONCE`.

Wrong/closed/drift/runtime-invalid authorization returns exit 2 before topology inspection, source read, recovery, loader call or filesystem mutation.

## Static preflight result and fixtures

The future preflight path is `scripts/streamq5_moe/preflight_het_next_l0_ph1_nvidia_nc2_compile_only.py`. Its exact native vector is `[base-alias,-I,-B,absolute-script,"--ack",preflight-ACK]`; `sys.argv` is `[absolute-script,"--ack",preflight-ACK]`.

The only positive artifact is `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc2_static_preflight_result.json`, capped at 1 MiB and atomically create-new via sibling `.inprogress.<pid>.<nonce16>`, file flush, write-through no-replace promotion and parent-directory flush. A valid repeat returns `already_complete`/exit 0 without rewriting. A stale/corrupt result is quarantined once and aborts/exit 3. Failure writes no positive artifact and may write exactly one bounded one-file attempt under `...nc2_static_preflight_failures`.

The result has exactly `kind="het_next_l0_ph1_nvidia_nc2_static_preflight_result"`, `revision="NC2"`, `status="pass"`, `pass=true`, `passed=16`, `total=16`, `check_names`, `checks`, `authorization`, `invocation`, `direct_bindings`, `fixture_manifest`, `fixture_results`, `transaction_results`, `exclusion_counters`, `artifact_bytes`. Exact ordered check names are: `direct_bindings`, `closed_to_open_authorization`, `runtime_identity`, `implementation_topology`, `core_import_architecture`, `create_operands`, `ctypes_abi`, `entrypoint_launch_separation`, `state_machine_fixtures`, `raw_nul_contract`, `ptx_cubin_parsers`, `toolchain_loader_contract`, `forbidden_io_and_calls`, `success_failure_transactions`, `independent_verifier_mutations`, `self_binding`.

The exact fixture manifest contains 84 unique names, grouped as follows:

- core (1): `success_log_nul_only`;
- API status (10): `status_version`, `status_create`, `status_compile`, `status_log_size`, `status_log_get`, `status_ptx_size`, `status_ptx_get`, `status_cubin_size`, `status_cubin_get`, `status_destroy`;
- API exception (10): the same names prefixed `exception_` instead of `status_`;
- state edges (6): `create_code0_null`, `create_nonzero_nonnull_destroy`, `compile_nonzero_log_success`, `compile_nonzero_logsize_secondary`, `compile_nonzero_logread_secondary`, `destroy_secondary_after_compile`;
- ABI (10): the ten API status suffixes prefixed `abi_`;
- raw bytes (8): `log_size_zero`, `log_one_non_nul`, `log_embedded_nul`, `log_missing_terminal`, `ptx_size_one`, `ptx_embedded_nul`, `ptx_missing_terminal`, `ptx_duplicate_terminal`;
- parser (9): `ptx_target_drift`, `ptx_address_drift`, `ptx_missing_entry`, `ptx_extra_entry`, `ptx_ftz`, `ptx_approx`, `ptx_unresolved`, `cubin_bad_elf`, `cubin_symbol_drift`;
- authorization/forbidden (8): `wrong_ack`, `closed_lock`, `hash_drift`, `extra_arg`, `imported_entry`, `wrong_interpreter`, `payload_open`, `driver_load`;
- transactions (12): `fresh_commit`, `valid_repeat`, `stale_temp`, `uncommitted_final`, `corrupt_commit`, `extra_final`, `prelink_failure`, `postlink_failure`, `file_fsync_failure`, `directory_fsync_failure`, `failure_writer_secondary`, `oversize`;
- production verifier (10): `valid_bundle`, `result_field`, `ledger_row`, `source_byte`, `log_byte`, `ptx_byte`, `cubin_byte`, `manifest`, `commit`, `topology`.

The manifest count is enforced from these literal names; every nonbaseline mutation must be rejected. Fixtures execute the exact future stdlib-only `compile_core` through an absolute/hash-bound `importlib.util.spec_from_file_location` bootstrap in an isolated `-I -B` child with injected fake DLL. The preflight never imports runner/backend/verifier and the child forbids real DLL, payload and filesystem calls.

## Isolated independent verification

The precommit verifier is a standalone script with no candidate import. After the runner has written and flushed the six-file staging tree, it launches exactly `[venv-python,"-I","-B",absolute-verifier,"--mode","precommit","--staging",absolute-staging,"--token","NC2_PRECOMMIT_VERIFY_BOUND_STAGING_V1"]`, with empty stdin, captured stdout/stderr, 60-second timeout and no shell. Parsed native/orig/sys/direct identity must satisfy the runtime contract. The verifier writes no file; stdout is exactly one UTF-8 JSON line with keys `kind="het_next_l0_ph1_nvidia_nc2_precommit_verification"`, `revision`, `mode="precommit"`, `pass`, `passed=14`, `total=14`, ordered `checks`, `staging_manifest_sha256`, `verifier_sha256`, `invocation`; exit is 0 only for 14/14, otherwise 3. The runner independently checks stdout schema/token-binding/hash/exit before promotion.

After commit, the verifier is separately invoked with exact `[venv-python,-I,-B,absolute-verifier,"--mode","postcommit","--bundle",absolute-output,"--ack",postcommit-ACK]`. It creates `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc2_independent_verification` containing exactly `verification.json`, `verification_manifest.json`, commit-last `verification_commit.json`, using the same durable create-new staging/promotion rules. Its ordered 14 checks are `direct_bindings`, `authorization`, `runtime_identity`, `bundle_topology`, `result_schema`, `source_create_operands`, `toolchain_modules`, `ledger_state_machine`, `log_canonical`, `ptx_contract`, `cubin_contract`, `manifest_commit`, `exclusions_resources_cleanup`, `terminal_contract`. `verification.json` has exactly `kind="het_next_l0_ph1_nvidia_nc2_independent_verification"`, `revision`, `status="positive"`, `pass=true`, `passed=14`, `total=14`, `check_names`, `checks`, `bundle_result_sha256`, `bundle_manifest_sha256`, `bundle_commit_sha256`, `authorization`, `invocation`, `mutation_results`, `artifact_bytes`. The verification manifest hashes that file; its commit binds manifest+result. Any failed check produces no positive verification tree and only a bounded verifier-failure attempt. The compile claim is independently verified only when this three-file verifier transaction exists and verifies.

## Topology classifier and recovery order

After authorization/runtime/hash validation but before mutation, one classifier enumerates literal family paths and returns exactly one state:

1. `valid_committed`: exact seven-file compile bundle with valid commit, no temp/orphan, and either absent or exact valid three-file verifier tree; return `already_complete`/exit 0 without mutation;
2. `fresh`: compile output, compiler failure root, quarantine root, all staging/temp globs and verifier output/failure/temp roots absent; only this state may compile;
3. `recoverable_stale`: exactly one bounded staging tree, uncommitted final, corrupt final or verifier-staging tree and no valid commit; move it once to a collision-free quarantine with disposition and abort/exit 3, never compile in the same invocation;
4. `invalid`: multiple states, extra/hidden file, oversize tree, failure history, quarantine collision or mixed success/failure. Up to four bounded entries may be quarantined in sorted order with one disposition; oversize is stat-only and moved without reading. If safe bounded quarantine is impossible, abort mutation-free. Never compile.

There is no later blanket “all paths absent” gate. Actual classifier/recovery TEMP tests cover fresh, valid repeat, every recoverable state, mixed/multiple/oversize/collision and wrong-authorization no-inspection/no-write.

## Exact evidence and terminal schemas

Nested success fields are closed:

- `authorization`: `ack_sha256`, `open_lock_path`, `open_lock_sha256`, `audit_token`, `authorized_qpc`, `invocation`;
- `source_identity`: `path`, `file_bytes`, `file_sha256`, `source_buffer_bytes=6174`, `source_buffer_sha256`, `program_name_ascii`, `program_name_buffer_sha256`, `num_headers=0`, `headers_null=true`, `include_names_null=true`;
- `toolchain_identity`: exact DLL/header/builtins path, size, SHA, version `[13,3]`, loader `CDLL`, `cdecl=true`, `winmode=4352`;
- each `runtime_modules` snapshot: `stage` in `before_load/after_compile/after_release`, `qpc`, ordered rows of `basename,resolved_path,bytes,sha256,system_module`;
- each ledger row: `sequence`, `operation`, `attempted`, `code`, `program_identity_before`, `program_identity_after`, `requested_bytes`, `returned_bytes`, `qpc_start`, `qpc_end`, `error`;
- `artifact_sizes`: exact integer keys `source,log,ptx,cubin,result,manifest,commit,total`;
- `exclusion_counters`: the six exact nonnegative integer counters preregistered above;
- six `resource_samples`, ordered `authorized,source_read,compiler_loaded,post_compile,post_destroy,post_serialize`, each `stage,qpc,rss,peak_wset,available_physical,disk_free,error`; all numeric values are integers, error is null, start available RAM >=2 GiB, disk >=128 MiB, peak working set <=1 GiB;
- `cleanup`: ordered rows `program_destroy`, `compiler_freelibrary`, `dll_directory_cookie_close`, each with `owned_before,attempted,code,qpc,error`; each owned resource is attempted once despite prior cleanup error;
- `filesystem_observation`: exact before/after paths, `compiler_cache_observed`, and unexpected mutations; any unexpected mutation is invalid.

Failure schema is the NC1 exact bounded schema plus these complete nested objects. A normal attempt contains only `failure.json`; if a failure occurs after canonical publication/flush, optional `postlink_secondary.json` is the only second file and has exactly `kind,revision,attempt_sha256,stage,operation,error_type,error,qpc,disposition,artifact_bytes`, cap 64 KiB. Failure to create that sidecar leaves the canonical attempt explicitly `postlink_unconfirmed`; it is invalid evidence and next invocation quarantines it. Primary errors never change.

Terminal classes are mutually exclusive: `compile_positive` (valid seven-file bundle), `compile_failure_valid_evidence` (exact bounded failure, never a pass), `infrastructure_invalid`, `writer_failure_invalid`, `already_complete`. Exactly one class is recorded; only `compile_positive` plus the later positive verifier transaction establishes the claim. Resource, authorization, protocol, lifecycle, cleanup, forbidden-call or topology failure is always infrastructure-invalid, never compiler-negative.

The DLL-directory cookie is registered immediately, stays alive through program destroy and compiler unload, then closes once. Cleanup order is program destroy, `_ctypes.FreeLibrary` of the registered NVRTC handle, cookie close, after-release module snapshot. Host exceptions remain attempted rows; later cleanup is still attempted. No deterministic repeat or compiler-cache claim is made.

## Durable compile/failure publication

The compile bundle remains exactly `result.json`, `source.cu`, `build.log`, `ptx.bin`, `cubin.bin`, `manifest.json`, commit-last `commit.json`. Manifest hashes the five data files; commit hashes those plus manifest. Staging, file flush, Windows directory-handle flush, write-through no-replace directory promotion and commit-last rules are unchanged. Caps remain 64 KiB source, 4 MiB log, 16 MiB PTX, 32 MiB CUBIN, 1 MiB per JSON, 56 MiB total. Failure JSON is capped at 1 MiB; postlink sidecar at 64 KiB. All writer, fsync, promotion, orphan, cleanup and collision transitions are exercised by actual future transaction functions in TEMP.

This design authorizes no implementation import, preflight, compiler, Driver or device call.

