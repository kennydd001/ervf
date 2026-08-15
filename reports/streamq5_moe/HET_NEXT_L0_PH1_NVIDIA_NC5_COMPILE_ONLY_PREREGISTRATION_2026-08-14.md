# PH1 NVIDIA NC5 compile-only preregistration

Status: **design-only; implementation, preflight, NVRTC and verification closed**.

NC5 can claim only that pinned NVRTC 13.3 accepted the exact frozen CUDA source under the frozen seven-option vector and returned canonical raw log/PTX/CUBIN bytes. It makes no numerical, performance, Driver/device, SASS, architecture-from-ELF or byte-repeatability claim.

The source remains `scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu`, 6,173 bytes/SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`, no NUL, exactly two entrypoints. Source buffer is only `ctypes.create_string_buffer(source_bytes)`, `sizeof==6174`, `.raw==source_bytes+b"\0"`, SHA `34f8f67c033061fc82866b5fe72c88d80c121b5b994dc4ce38d27aa4a0cc3c47`. Program name is only `ctypes.create_string_buffer(b"het_next_l0_ph1_nvidia_nc5_kernels.cu")`, `sizeof==38`, exact single terminal NUL, SHA `f79ea9c0c7cb56bee5f4082cbd7c9a7609db9ffd7aa3526fb38df31d55c3d5c7`. Headers count is zero and both header pointers are typed NULL.

One direct `.venv\Scripts\python.exe -I -B` runner process performs authorization, NVRTC calls, explicit unload and publication. There is no compiler child, IPC or imported entry. A later verifier is a separate CPU-only process after runner termination; a recovered `.inprogress` phase retry is a new separately authorized process, never an in-process retry.

Raw log/PTX each have exactly one terminal NUL/no embedded NUL; parsers see strict UTF-8 `raw[:-1]`; hashes include NUL. PTX requires `.target sm_120`, `.address_size 64`, and exactly `q5_linear`,`bf16_lut_activation`. CUBIN is bounded valid ELF64 with the two kernel symbols.

Successful `result.json` keys are exactly `kind,revision,status,terminal_class,claim,authorization,invocation,source_identity,create_operands,toolchain_identity,runtime_modules,options,entrypoints,future_launch_uses,ledger,artifacts,exclusion_counters,resource_samples,filesystem_observation,ownership,cleanup`. `artifact_sizes` is forbidden. `artifacts` has exactly `source,build_log,ptx,cubin,disassembly`; the first four contain only `bytes,sha256`, disassembly is `status="not_attempted_out_of_scope",bytes=0,sha256=null`. Final in-result resource stage is `pre_result_serialize`. Manifest is computed afterward from the five immutable data/result files; commit binds manifest and is last.

Terminal transactions are immutable and mutually exclusive: `compile_positive`; `compile_valid_negative`; `incidental_failure`; `already_complete`. A valid negative retains `source.cu` and every bounded raw log/PTX/CUBIN actually observed. Missing artifacts are explicit `not_attempted` or `unavailable_after_error` fields in `negative.json`. Over-cap evidence retains the first 65,536 bytes plus metadata containing exact full size/SHA and prefix size/SHA. Its manifest hashes every retained file, then commit is last. It is never retried or quarantined.

A correlated postlink incident is ancillary: it never consumes/reopens the committed attempt and never changes bytes. Topology evaluates the exact correlated incident before returning the immutable compile/verifier terminal; an exact valid compile remains `already_complete`, while the ancillary incident remains separately visible. Mismatched/unexplained incidents are invalid.

Only exact `.inprogress` debris is recoverable: `same_invocation_retry=false`, `attempt_consumed=false`, `next_invocation_allowed=true`. Every positive, valid-negative, verifier-protocol-negative, already-complete or incidental terminal has `same_invocation_retry=false,next_invocation_allowed=false`.

No implementation or execution is authorized.

