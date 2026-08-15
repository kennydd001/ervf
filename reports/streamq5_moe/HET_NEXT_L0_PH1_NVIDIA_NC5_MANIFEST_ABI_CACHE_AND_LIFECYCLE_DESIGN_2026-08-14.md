# NC5 manifest, ABI, cache and lifecycle design

## Normative matrix

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc5_fixture_manifest.json` is normative. It has 294/294 unique cases and literal fake source/log/PTX bytes plus a literal valid 536-byte ELF. Canonical PTX decodes to exactly 130 bytes/SHA `3b4cde8b9803cd2dd6131ac2776730915a5f2b3c5f17c9b690c08db6143f4336`, exactly one final NUL and two `.visible` entries. ELF is exactly 536 bytes/SHA `93abe3a2a7c4f7b4e6b6b9ce202ecc9440a02c3d37a9b9e8f476939d102cf2c8`.

Every case records injection, all ten expected ledger rows, tagged primary union `{state:"none",value:null}` or `{state:"failure",value:<nonempty string>}`, secondary failures, disposition, evidence files and negative artifact states. All 292 attempted `nvrtcCreateProgram` rows have handle-before zero; handle-after is exact zero/H1. Valid repeats have no primary and `publish="no_write_existing_terminal"`. The manifest directly specifies stage-dependent raw negative bundles; generic inheritance is forbidden.

Preflight stores only compact manifest identity `{path,bytes,sha256,count:294}` and compact results `[name,pass,observed_digest]`, never embeds the manifest/ledgers. Exact caps: fixture manifest, preflight result, preflight failure and every JSON are 4,194,304 bytes; source 65,536; log 4,194,304; PTX 16,777,216; CUBIN 33,554,432; complete bundle 67,108,864. Preflight positive/failure/quarantine, compile positive/negative/incidental/quarantine, and verifier positive/protocol-negative/incidental/quarantine roots are the exact NC4 paths with `nc4` replaced by `nc5`. Each phase uses create-new `.inprogress.<pid>.<nonce16>`, file/directory flush, write-through no-replace promotion and commit-last. Only exact inprogress debris is quarantined; terminals/failure history are not debris.

## Exact Win64 loader ABI

`kernel32=ctypes.WinDLL("kernel32",use_last_error=True)`. Aliases are `BOOL=c_int32`, `DWORD=c_uint32`, `HMODULE=c_void_p`, `FARPROC=c_void_p`, `DLL_DIRECTORY_COOKIE=c_void_p`, `LPCWSTR=c_wchar_p`, `LPCSTR=c_char_p`, `HANDLE=c_void_p`. Exact signatures:

| function | argtypes | restype |
|---|---|---|
| `AddDllDirectory` | `[LPCWSTR]` | `DLL_DIRECTORY_COOKIE` |
| `RemoveDllDirectory` | `[DLL_DIRECTORY_COOKIE]` | `BOOL` |
| `LoadLibraryExW` | `[LPCWSTR,HANDLE,DWORD]` | `HMODULE` |
| `FreeLibrary` | `[HMODULE]` | `BOOL` |
| `GetProcAddress` | `[HMODULE,LPCSTR]` | `FARPROC` |
| `GetModuleHandleW` | `[LPCWSTR]` | `HMODULE` |

Load flags are exactly `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR=0x100`, `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS=0x1000`, combined `0x1100`; reserved handle is NULL. Before every Win32 call `ctypes.set_last_error(0)`; on NULL/false, capture `ctypes.get_last_error()` immediately before any other call. On success, last-error is diagnostic only. Handles/function addresses must round-trip as unsigned pointer-width nonzero integers. Static mutations change every argtype/restype, calling convention, flag, NULL/non-NULL transition and last-error capture order and must fail before loading a real DLL.

The runner never creates an owning `ctypes.CDLL`; it uses `GetProcAddress` plus cdecl `CFUNCTYPE` NVRTC wrappers. Cookie/HMODULE register immediately. One `finally` attempts program destroy, poisons/discards wrappers, calls `FreeLibrary` exactly once, poisons the handle regardless of return, calls `RemoveDllDirectory` exactly once, then checks both modules absent. All cleanup rows remain attempted despite earlier failure; no post-free wrapper use or second release exists.

## Cache and filesystem policy

After authorization, create one fresh private redirected tree under the phase staging root. Set `CUDA_CACHE_DISABLE=1`, `CUDA_CACHE_MAXSIZE=0`, and set `CUDA_CACHE_PATH`, `TMP`, `TEMP`, `NVRTC_CACHE_PATH` to exact subdirectories. The tree may contain transient or retained compiler files; emptiness is not a gate. Before load, after every NVRTC call, after unload and precommit, recursively retain a canonical manifest of each private path: relative path, type, bytes, mtime_ns, SHA for files. Final private-tree contents are retained inside the evidence bundle or negative bundle and hashed by its manifest, subject to total cap. Any write outside the exact staging/private tree is incidental-invalid. No claim is made that after-call snapshots detect already-deleted transient files; containment is enforced by the redirected environment and external-root before/after manifests.

## Topology and evidence priority

Authorization precedes topology inspection/mutation. Literal priority is: correlated postlink incident; exact positive/valid-negative/verifier-negative terminal; exact one inprogress debris; fresh; invalid. Correlated ancillary evidence must name phase, terminal commit SHA, originating attempt nonce, postlink operation and immutable terminal class. Exact correlation preserves terminal `already_complete`; unexplained correlation invalidates. Mixed/multiple terminals, extra/hidden roots, mismatched incidents, non-inprogress partials, collisions or oversize are invalid and mutation-free.

Static preflight executes the exact manifest against fake APIs and actual future transaction/classifier functions only. No NVRTC, Driver, device or scientific payload call is permitted. This design is closed.

