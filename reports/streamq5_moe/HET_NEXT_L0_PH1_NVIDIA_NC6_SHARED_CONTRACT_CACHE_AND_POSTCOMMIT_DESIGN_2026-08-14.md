# NC6 shared contract, cache and postcommit design

## Shared production functions

Future `scripts/streamq5_moe/het_next_l0_ph1_nvidia_nc6_compile_contract.py` is stdlib-only and import-inert. Top level defines constants/types/functions only; importing performs no authorization, file inspection/mutation, environment mutation, NVRTC/Win32 load, payload or device action. It exports exactly `classify_topology`, `recover_inprogress`, `publish_transaction`, `write_incidental_failure`, `adjudicate_terminal`.

The runner and preflight import this exact hash-bound module. The runner uses those exports for physical topology/publication. Preflight injects a TEMP filesystem and fake API into those same function objects for all 297 manifest cases. Source/AST gates reject local copied classifiers, wrappers returning snapshots, monkeypatched replacement functions, different `__code__` hashes, or imports from an ancestor revision. The preflight records module absolute path/SHA and each export's qualified name/code digest and proves runner/preflight identities equal. This shared module contains no NVRTC call; the runner alone supplies the physical operation callbacks after authorization.

## Postcommit incident and topology

A failure after commit-link/promotion is written only to the phase incidental-failure root with exact fields `kind,revision,status="postcommit_incident",phase,operation,primary_error,secondary_errors,terminal_commit_path,terminal_commit_sha256,originating_nonce,device_opened=false,driver_loaded=false,cleanup,dispositions,artifact_bytes`. It does not modify, consume, append to, quarantine or reopen the committed transaction. Topology priority is `postcommit_incident -> immutable terminal -> exact inprogress debris -> fresh -> invalid`; encountering the incident returns exit 3 and forbids runner/preflight direct retry.

A separately authorized CPU verifier may read the immutable commit and exactly correlated incident. It produces a new immutable `durability_adjudication` transaction with result/manifest/commit. Until that transaction is positive, the original compile claim is not upgraded to positive/already-complete. Mismatched, multiple or unexplained incidents are invalid. The compiler is never rerun.

Only an exact one-pattern `.inprogress.<pid>.<nonce16>` subtree without a terminal is `transaction_debris`. Recovery moves only that subtree to the phase quarantine, writes exact disposition, aborts current invocation, and permits a later authorized process. Any published failure, positive/negative terminal, non-inprogress partial, multiple/mixed entry, collision or oversize is terminal/invalid and not retryable.

## Cache containment and bundles

After authorization, create one fresh private tree under staging and redirect `CUDA_CACHE_PATH`, `TMP`, `TEMP`, `NVRTC_CACHE_PATH` to exact subdirectories while setting `CUDA_CACHE_DISABLE=1`, `CUDA_CACHE_MAXSIZE=0`. The tree may be nonempty. Snapshot before load, after every NVRTC call, after unload and precommit. The final canonical observation is embedded in result/negative JSON and hashed normally; its actual private files remain inside exact `private_tree/` in the promoted positive or negative bundle. The manifest enumerates every directory/file beneath that prefix with size/SHA and includes total bytes in the 67,108,864-byte bundle cap. Reparse points, symlinks, `..`, absolute paths and external writes are forbidden. Cleanup never deletes retained final private-tree bytes; only transient inprogress debris is recoverable.

Executable cache fixtures use actual shared transaction/classifier functions: empty tree pass, one deterministic retained `private_tree/nvrtc/cache.bin` pass with exact observation, and `../outside.bin` rejection. Manifest and verifier mutations cover entry order/path/type/size/mtime/SHA/tree digest, missing/extra retained file, traversal and external write.

## Caps and exact phase evidence

Fixture manifest, preflight result/failure and every JSON cap are 4,194,304 bytes; source 65,536; log 4,194,304; PTX 16,777,216; CUBIN 33,554,432; embedded prefix 4,096; total terminal bundle 67,108,864. Static preflight stores only manifest identity `{path,bytes,sha256,count:297}` plus compact rows `{name,pass,observed_digest}`.

Positive/negative/failure schemas and exact Win64 ABI remain NC5 except the added cache-tree observation/files and `postcommit_incident` schema above. Result has no `artifact_sizes`. Failure bundles never use a prefix sidecar. All canonical files are create-new, fsynced, directory-flushed, no-replace promoted and commit-last where terminal.

This design is closed; no function import or execution is authorized yet.

