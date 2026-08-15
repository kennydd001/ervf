# HET-NEXT-L0-PH1 Intel compile R0

Date: 2026-08-14  
State: source frozen; execution closed pending source audit and static preflight.

This phase compiles only the PH1 Intel OpenCL source for the exact proven Intel Arc device. It may enumerate the OpenCL platform/device, create a context and program, build, retain source/build-log/binary evidence, then release program and context. It must create no queue, kernel, event, USM allocation, buffer or payload object and must launch no kernel. It must not open the official shard, D2 raw, PH1 records, LUT or CPU safetensors.

Eligibility binds:

- committed PH1 CPU package commit SHA-256 `f3677e9610bea03649fec172b97c0c314f2f2e4c0d40bf9d864df0ec88a44f06`;
- independent CPU verification JSON SHA-256 `1c7f2772fb637485020be00f74b6f9295a18ec3d7d10af0587ea350e8756cbc8` and `pass=true`;
- physical contract SHA-256 `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981`;
- context addendum SHA-256 `dde29c369c5218f5cca3ed12248979a8c03c95b51e8b433f65175750d74d695c`;
- exact backend and OpenCL source hashes from the execution lock.

The source must define exactly four entrypoints and the frozen gate/up/down reduction DAG plus integer BF16 LUT activation. Build options are exactly `-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt`; fast-relaxed-math, finite-math-only, unsafe-math, mad-enable and denorm-flush options are forbidden. The static preflight parses the actual Python AST, extracts the source literal, verifies entrypoint/cardinality/buffer/launch constants, independently exercises all BF16 multiply vectors and negative source mutations, and contains no OpenCL/device/payload call.

On an authorized compile, retain exact source bytes, build-log bytes, program binary bytes, identity, options, ordered ledger and cleanup. A positive requires exact Intel name/vendor/driver/PCI/USM extension; successful build; nonempty binary; zero payload reads, allocations, kernels and launches; release attempts for program and context; cleanup complete. Output is create-new temp, fsynced, exact-manifested, commit-last and promoted write-through. A failure retains immutable cleanup evidence. One authorized compile attempt only; no retune or retry. This phase opens no correctness/device-execution claim.
