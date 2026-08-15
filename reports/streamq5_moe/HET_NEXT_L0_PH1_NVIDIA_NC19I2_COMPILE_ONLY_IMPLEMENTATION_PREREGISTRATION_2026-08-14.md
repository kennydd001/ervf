# PH1 NVIDIA NC19I2 compile-only implementation preregistration

NC19I2 is an immutable successor to NC19I1 and binds its independent NO-GO source audit (`578df903f3dd975c41d3eeb68b8f70674ebd4d5f8f85fc180b6ce791ae957ab5`). It preserves the NC19 design boundary: one 6,173-byte CUDA source, one NVRTC 13.3 program, ten ordered NVRTC calls, build log, PTX, cubin, and no scientific payload, nvcuda/Driver, CUDA Runtime, context, module, allocation, copy, launch, or device operation.

The runner authenticates exact direct `.venv` CPython `-I -B` invocation, closed/open locks, preflight result, source/toolchain identities, and runtime topology before mutation or compiler loading. The mandatory preflight result is a runtime input; compile positive and compile-valid-negative are exclusive terminals. The runner calls the shared production topology classifier and terminal adjudicator. Invalid authorization is mutation-free.

The source/name buffers have one terminal NUL. Artifact sizes are checked before allocation. The cache tree is stat-capped before bounded streaming reads. Module evidence distinguishes pre-load, after-load, during-compile (NVRTC plus builtins), and post-release. Every owned program/module/cookie identity and cleanup return is cross-linked. Environment capture, four private cache paths, twelve ordered snapshots, and reverse restoration are retained.

All publications are create-new and durable. Failures after hard-link or rename roll back the canonical target before returning; tests inject pre-link, post-link, precommit, post-rename, and failure-writer faults. Only in-progress debris is recoverable. Positive, real compile-negative, and mutations are independently checked.

Static preflight remains closed. This freeze authorizes only later no-device preflight after audit. No preflight, NVRTC/compiler, payload, Driver, runtime, or device call was made while freezing this package.
