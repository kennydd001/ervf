# PH1 NVIDIA NC3 compile-only preregistration

Date: 2026-08-14  
Status: **design frozen; implementation and every execution phase closed**.

## Claim, source and exact create input

NC3 asks only whether pinned NVRTC 13.3 compiles one exact CUDA source into canonical raw log, PTX and CUBIN artifacts. It makes no numerical, performance, device, Driver, architecture-from-ELF, disassembly or byte-repeatability claim.

The sole compiler source is `scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu`, 6,173 bytes, SHA-256 `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781`, no NUL, with exactly two entrypoints: `q5_linear`, `bf16_lut_activation`. Four later launch labels remain out of scope.

Source construction is exactly `source_buffer=ctypes.create_string_buffer(source_bytes)`, never an already terminated initializer. Gates require `ctypes.sizeof(source_buffer)==6174` and `source_buffer.raw==source_bytes+b"\0"`; its SHA-256 is `34f8f67c033061fc82866b5fe72c88d80c121b5b994dc4ce38d27aa4a0cc3c47`. Program-name construction is exactly `ctypes.create_string_buffer(b"het_next_l0_ph1_nvidia_nc3_kernels.cu")`; `sizeof==38`, `.raw==b"het_next_l0_ph1_nvidia_nc3_kernels.cu\0"`, SHA-256 `78416327c270a471f60289892d406e5e7f145d44e8e7288eb50759dfb1e2c890`. `numHeaders=0`; headers and includeNames are typed NULL `POINTER(c_char_p)()`.

The seven ordered options and complete ten-function cdecl ABI table are exactly those frozen in NC2. All restypes are `c_int`; aliases and argument vectors are unchanged. The ten-row first-primary/later-secondary state machine, including compile-error log retrieval and mandatory reverse destroy for any nonnull program, is unchanged.

## Raw artifacts and noncircular result

Raw log and PTX sizes equal their NVRTC size rows, are greater than/equal to one and greater than one respectively, have exactly one terminal NUL and no embedded NUL, and are digested including the NUL. Logical strict UTF-8 parsers see only `raw[:-1]`. PTX requires `.version`, `.target sm_120`, `.address_size 64`, exactly the two entrypoints, and no FTZ/approximate/fast-math/unresolved function evidence. CUBIN is >1 byte, bounded ELF64 with valid tables and exactly the two named CUDA kernel symbols among kernel entries.

Before result serialization, `result.json.artifacts` contains exactly five rows/keys: `source`, `build_log`, `ptx`, `cubin`, `disassembly`. The first four have only `bytes:int` and `sha256:str`; disassembly is exactly `{status:"not_attempted_out_of_scope",bytes:0,sha256:null}`. No result/manifest/commit/total self-size occurs in result. The final in-result resource stage is truthfully `pre_result_serialize`; postserialization/postcommit facts belong only to later manifest/commit/verifier evidence.

`manifest.json` is computed after immutable data files and result, and hashes ordered `source.cu`, `build.log`, `ptx.bin`, `cubin.bin`, `result.json`. `commit.json` is computed last and binds the manifest SHA plus those five hashes. The compile bundle contains exactly those seven files. Canonical JSON is UTF-8, sorted keys, compact separators, one LF.

## Toolchain and exclusions

Pinned identities remain NVRTC DLL SHA `c7af6b5d…b05fe`, builtins SHA `82c70380…b2733a`, header SHA `316a1375…72d21`, version `[13,3]`. No official shard, D2, CPU stage/LUT, model/tokenizer, Torch, Transformers, Safetensors, CuPy, cudart, nvcuda, Driver/device/context/module/memory/copy/launch/sync or physical artifact is permitted.

No implementation, preflight, compiler, Driver or device call is authorized.

