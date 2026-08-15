# PH1 NVIDIA NC4 compile-only preregistration

Status: **design-only; all implementation and execution closed**.

NC4 makes one narrow claim: pinned NVRTC 13.3 accepted the exact frozen CUDA source under the exact seven NC2 options and returned canonical log/PTX/CUBIN bytes. It makes no numerical, performance, Driver, device, SASS, architecture-from-ELF, or repeatability claim.

The source remains `scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu`, 6,173 bytes/SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`, exactly two entrypoints. Source construction is only `ctypes.create_string_buffer(source_bytes)` and must satisfy `sizeof==6174`, `.raw==source_bytes+b"\0"`, SHA `34f8f67c033061fc82866b5fe72c88d80c121b5b994dc4ce38d27aa4a0cc3c47`. Program-name construction is only `ctypes.create_string_buffer(b"het_next_l0_ph1_nvidia_nc4_kernels.cu")` and must satisfy `sizeof==38`, exact single-NUL raw buffer, SHA `696ec3beaa93b373e994bf16e96319e2e0e32828eac34fefa821d280fab39453`. `numHeaders=0`, headers/includeNames typed NULL. The NC2 ten-function cdecl ABI and ten-row state machine are unchanged.

Exactly one `.venv\Scripts\python.exe -I -B <absolute-runner> --ack <exact-token>` process performs authorization, direct NVRTC calls, artifact construction, explicit unload and publication. There is no compiler child, IPC, worker, subprocess or imported entry. A later independent verifier is a separate CPU-only invocation after compile termination.

Raw log and PTX include exactly one terminal NUL/no embedded NUL; logical strict UTF-8 parsing uses `raw[:-1]`; hashes cover raw bytes. PTX requires target `sm_120`, address 64 and exactly the two entries. CUBIN is bounded valid ELF64 with both kernel symbols.

The exact successful `result.json` top-level key set is `kind,revision,status,terminal_class,claim,authorization,invocation,source_identity,create_operands,toolchain_identity,runtime_modules,options,entrypoints,future_launch_uses,ledger,artifacts,exclusion_counters,resource_samples,filesystem_observation,ownership,cleanup`. No `artifact_sizes` key exists. `artifacts` has exactly `source,build_log,ptx,cubin,disassembly`; the first four contain only `bytes,sha256`, while disassembly is exactly `status="not_attempted_out_of_scope",bytes=0,sha256=null`. The last result resource stage is `pre_result_serialize`. Manifest is built afterward from `source.cu,build.log,ptx.bin,cubin.bin,result.json`; commit binds the manifest and is last.

Terminal states are mutually exclusive and immutable:

- `compile_positive`: exact committed seven-file bundle, exit 0; no compile retry;
- `compile_valid_negative`: exact committed three-file `negative.json,negative_manifest.json,negative_commit.json` for an NVRTC compile/retrieval/artifact outcome with complete cleanup, exit 3; no retry or quarantine;
- `incidental_failure`: exact bounded phase failure attempt for authorization-excluded infrastructure/protocol/resource/lifecycle/writer faults, exit 3 or writer exit 4; no reclassification;
- `already_complete`: exact existing positive or valid-negative transaction, exit 0 without mutation.

The independent verifier similarly commits either a positive three-file verification transaction or immutable `verifier_protocol_negative` three-file transaction; neither is quarantined or retried. Only exact `.inprogress` debris is recoverable.

Compiler caching is disabled by `CUDA_CACHE_DISABLE=1`, `CUDA_CACHE_MAXSIZE=0`. After authorization the runner creates one fresh private empty cache/temp directory and sets `CUDA_CACHE_PATH`, `TMP`, `TEMP` and `NVRTC_CACHE_PATH` to it. It snapshots after each NVRTC call. Any child entry/file mutation is incidental-invalid; only creation/removal of the empty directory is allowed.

No implementation, preflight, compiler, Driver or device call is authorized.

