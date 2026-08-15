# PH1 NVIDIA NC6 compile-only preregistration

Status: **design-only; implementation and all execution closed**.

NC6 retains the NC5 narrow claim, exact source/name buffer construction, seven NVRTC options, ten-call ABI/state machine, canonical 130-byte PTX and 536-byte ELF fixture, noncircular result keyset, raw compiler-negative evidence, direct one-shot `.venv -I -B` runner, exact Win64 loader ABI and explicit unload. It makes no numerical, performance, Driver/device, SASS, architecture-from-ELF or repeatability claim.

The sole normative mutation/evidence matrix is `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc6_fixture_manifest.json`. It has 297 literal cases: the corrected 294 NC5 cases plus `cache_empty_private_tree`, `cache_private_file_retained`, `cache_external_write_rejected`. All attempted create rows begin with handle zero. Every primary uses the tagged none/failure union.

Terminal dispositions are exact:

- `compile_positive`, `compile_valid_negative`, `verifier_protocol_negative`, `incidental_failure`, and `already_complete` are terminal: `same_invocation_retry=false,next_invocation_allowed=false`;
- `postcommit_incident` is terminal, exit 3, publishes a separate failure attempt, never reopens/consumes/reclassifies the immutable commit, and forbids direct rerun; only a later separately authorized CPU verifier may adjudicate the existing commit plus incident;
- `transaction_debris` is the only nonterminal: it denotes exactly one `.inprogress` subtree, publishes only its quarantine disposition, has `attempt_consumed=false,next_invocation_allowed=true`, and permits only a later clean process invocation.

Every incidental-failure row is terminal and has `next_invocation_allowed=false`. Every other state marked next-allowed is invalid.

Over-cap negative evidence is self-contained in `negative.json`, never a guessed/null prefix file. The exact artifact object is `{state:"bounded_prefix_embedded",embedded_prefix:{source,offset:0,length:min(4096,available_bytes),base64,sha256,derivation}}`. The injected over-cap stream is the exact baseline raw artifact repeated cyclically to its cap plus one byte; the embedded prefix is therefore independently reconstructible. Retained source and all other observed bounded raw artifacts remain ordinary manifest-hashed files; missing artifacts have only explicit `not_attempted` or `unavailable_after_error` state.

The private redirected compiler tree is evidence, not required empty. `result.json.filesystem_observation.cache_tree` has exactly `private_root,before,after,tree_digest`. Each snapshot is `qpc,entries,tree_digest`; sorted entries are exactly `path,type,size,mtime_ns,sha256`. Tree digest is SHA-256 of ordinal entries serialized as UTF-8 `path NUL type NUL decimal_size NUL decimal_mtime_ns NUL sha256 LF`. Files are allowed only below the private root. Any external write is incidental-invalid. Empty and retained-file cases must pass and be retained; external-write must fail.

No implementation, preflight, NVRTC, Driver or device call is authorized.

