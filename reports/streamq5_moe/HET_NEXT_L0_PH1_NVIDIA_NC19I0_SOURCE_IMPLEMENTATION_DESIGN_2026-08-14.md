# NC19I0 source implementation design

Four substantive sources are frozen: shared `compile_contract`, direct-NVRTC runner, static no-device/no-payload preflight, and independent verifier. The source lock directly binds them, the immutable N5 CUDA source, NC19 manifest/design/audit, Python/NVRTC/builtins/header identities, and exact closed output topology.

The compile contract is import-inert and stdlib-only. Compile uses kernel32 `WinDLL`, `AddDllDirectory`, `LoadLibraryExW(0x1100)`, `GetProcAddress`, cdecl `CFUNCTYPE`, one program, and release-all cleanup. Source/name buffers have exactly one terminal NUL. The runner redirects six environment variables to four distinct private directories, records pre-load plus every operation and post-release snapshots, restores in reverse order, and promotes only after the standalone verifier passes.

The static preflight uses fake APIs and temporary files only. It validates every NC19 manifest case and raw observation, exact source-lock 100/57/0 invariants, structural kernel/AST constraints, shared function identities, compile fault suffixes, environment combinations, topology, transactions, and verifier contract shape. This document authorizes no execution.
