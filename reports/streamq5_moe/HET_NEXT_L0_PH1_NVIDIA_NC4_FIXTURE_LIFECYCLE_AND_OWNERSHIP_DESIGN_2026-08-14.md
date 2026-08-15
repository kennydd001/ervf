# NC4 fixture, lifecycle and ownership design

## Literal executable fixture contract

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc4_fixture_manifest.json` is normative. It contains 294 unique cases. Each case records exact injection kind/op/index/code/exception/nonnull sentinel/input mutation, all ten expected ledger rows, primary and ordered secondary failures, terminal, exit, publication and retry policy. It embeds the exact fake source/log/PTX bytes and the complete literal base64 for a 536-byte EM_CUDA ELF64 fixture, together with the independently reconstructable header, offsets, five sections, strings and three symbols. The literal ELF SHA is `93abe3a2a7c4f7b4e6b6b9ce202ecc9440a02c3d37a9b9e8f476939d102cf2c8`; reconstruction must equal the literal bytes exactly. Static preflight must exact-set/equality check every field; snapshot-only or generic group booleans are forbidden.

## Paths and transactions

Phase roots are literal:

- preflight result `...nc4_static_preflight_result.json`, failure `...nc4_static_preflight_failures`, quarantine `...nc4_static_preflight_quarantine`;
- compile positive `...nc4_compile_only`, compile negative `...nc4_compile_only_negative`, incidental failure `...nc4_compile_only_incidental_failures`, quarantine `...nc4_compile_only_quarantine`;
- verifier positive `...nc4_independent_verification`, verifier negative `...nc4_independent_verification_negative`, verifier incidental failure `...nc4_independent_verification_failures`, quarantine `...nc4_independent_verification_quarantine`.

Positive compile schema is the exact NC4 result key set plus canonical five-row manifest and commit. Negative schema is `kind,revision,status="compile_valid_negative",stage,primary_error,secondary_errors,authorization,invocation,source_identity,toolchain_identity,runtime_modules,options,ledger,artifacts,exclusion_counters,resource_samples,filesystem_observation,ownership,cleanup,terminal_class`; negative manifest has `kind,revision,canonical_json,files=[negative.json]`; negative commit has `kind,revision,state="complete",manifest_sha256,negative_sha256,committed_qpc`.

Incidental failure attempt contains exactly `failure.json`, with `kind,revision,status="incidental_failure",phase,stage,error_type,error,primary_error,secondary_errors,authorization_state,device_opened=false,driver_loaded=false,compiler_loaded,payload_bytes_read,ledger,runtime_modules,resource_samples,ownership,cleanup,filesystem_observation,dispositions,artifact_bytes,correlated_commit_sha256_or_null`. A postlink error is represented only as a new create-new incidental attempt referencing the already immutable commit; no committed file/sidecar is modified. If the incident writer fails, its primary remains the original and writer fault is secondary in a separate bounded writer-failure attempt if writable; otherwise no false valid evidence is claimed.

Verifier positive/negative result fields are exactly `kind,revision,status,terminal_class,pass,passed,total,check_names,checks,compile_result_sha256,compile_manifest_sha256,compile_commit_sha256,authorization,invocation,mutation_manifest_sha256,mutation_results,artifact_bytes`. Their manifest is exactly `kind,revision,canonical_json,files` with the one result row; commit is exactly `kind,revision,state,manifest_sha256,result_sha256,committed_qpc`. Positive uses pass true/14/14; protocol-negative uses pass false, exact false-check set and immutable exit 3. Phase failure/disposition fields are identical to compile incidental schema with `phase` fixed.

Every transaction uses create-new sibling `.inprogress.<pid>.<nonce16>`, file flush, Windows directory-handle flush, no-replace write-through promotion, commit last, and caps from NC3. Only an exact single `.inprogress` subtree with no committed terminal is recoverable: move it once to its same-phase quarantine with `disposition.json` (`kind,revision,phase,source,target,reason,size,sha256_or_null,qpc,move_code`) and abort. Corrupt committed, multiple, mixed positive/negative, extra/hidden, collision or oversize is invalid and mutation-free. A valid compile plus verifier `.inprogress` quarantines only verifier debris; compile bytes never move. Valid terminal/failure records are never recovery debris.

## Allowed I/O and exact topology

Authorization/runtime/hash checks precede inspection/mutation. Preauthorization application reads are only current script/open lock/interpreter identities/direct small authorization bindings; writes none. Authorized reads are those plus frozen implementation files, fixture manifest, CUDA source, NVRTC DLL/builtins/header, and the active evidence tree. Scientific payload is never allowed. Writes are only the active phase roots and private empty cache directory. The common guard increments attempted/read/written counters before access. Runtime module and filesystem snapshots use exact path/type/size/mtime/hash fields. Any nonallowlisted module, read, write or cache file is incidental-invalid.

Topology classification order is `authorized -> literal enumerate -> exact committed terminal -> exact one inprogress -> fresh -> invalid`. Exact allowed entries are design/source/locks/audits, at most one immutable compile terminal, at most one immutable verifier terminal, phase failure histories, and no unexplained item. Multiple compile terminals, multiple verifier terminals, positive+negative mixtures, commit+uncommitted, orphan, hidden/temp outside the exact pattern, or cap excess are invalid without cleanup.

## Same-process DLL ownership

The one runner calls ABI-bound `AddDllDirectory`, `LoadLibraryExW(absolute_nvrtc,NULL,0x1100)`, `GetProcAddress`, `FreeLibrary`, `RemoveDllDirectory`, `GetModuleHandleW`. It never constructs an owning `ctypes.CDLL`. It creates cdecl `CFUNCTYPE` wrappers from export addresses. Cookie and HMODULE are registered immediately on nonnull return. Before-load requires NVRTC/builtins absent. In one `finally`, it attempts program destroy, discards every wrapper, calls `FreeLibrary` exactly once, nulls/poisons the stored handle regardless of code, calls `RemoveDllDirectory` exactly once, then snapshots modules. No wrapper/use/release occurs after FreeLibrary.

Cleanup rows are exactly `program_destroy,wrapper_discard,compiler_freelibrary,dll_directory_remove,postrelease_module_check`, each with `sequence,resource,identity,owned_before,attempted,code,qpc,error`. All owned rows are attempted despite earlier errors. Success/valid-negative requires destroy agreement with ledger, FreeLibrary and RemoveDllDirectory success, and both NVRTC/builtins absent after release. Failure leaves terminal lifecycle-invalid; process exit is only ultimate OS cleanup and never upgrades evidence.

Static preflight exercises all 294 cases against the exact production state machine and transaction functions with fake APIs only. It performs no NVRTC, Driver, device or payload call. This design itself authorizes no preflight.
