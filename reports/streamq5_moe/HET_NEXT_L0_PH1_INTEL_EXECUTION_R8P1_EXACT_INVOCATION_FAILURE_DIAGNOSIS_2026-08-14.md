# PH1 Intel execution R8P1 — exact-invocation failure diagnosis

Date: 2026-08-14  
Scope: read-only diagnosis of the single authorized R8P1 attempt. No rerun, candidate import, payload read, compiler, OpenCL, or device call was performed.

## Verdict

R8P1 is an immutable **protocol-negative** attempt caused by an incorrect Windows-venv invocation model. It is not a numerical, CPU-preparation, compiler, OpenCL, or device failure.

The observed process stopped at `preflight_het_next_l0_ph1_intel_execution_r8p1.py:188` with `RuntimeError("exact_invocation")`, exit code 1, after approximately 0.128 seconds. This is before `collect_runtime()` at line 191, `preparation_summary()` at line 194, and publication at line 197.

## Exact failed conjuncts

The frozen R8P1 expectation used the venv launcher as element zero of all four process views:

`C:\Users\de_do\Documents\ChatGPT\New project\.venv\Scripts\python.exe`

The command did launch that file. Its immutable SHA-256 is `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`, and inside the process `sys.executable` remained that exact venv path.

On this Windows Store CPython venv, however, the launcher transfers control to the base interpreter named by `.venv\pyvenv.cfg`:

`C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe`

The base path is consequently element zero of `GetCommandLineW`, its `CommandLineToArgvW` parse, and `sys.orig_argv`. This is consistent with the pre-existing `pyvenv.cfg`, SHA-256 `9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe`, whose frozen `home`, `executable`, and `command` fields all name that WindowsApps installation.

Therefore exactly these R8P1 comparisons fail:

1. raw `GetCommandLineW` text versus a command rendered with the venv path at argv[0];
2. parsed Win32 argv versus `EXPECTED_ORIG` at element zero;
3. `sys.orig_argv` versus `EXPECTED_ORIG` at element zero;
4. `resolved_executable` derived from `sys.orig_argv[0]` versus the venv launcher path.

The application vector (`sys.argv`), exact script, `--ack`, token, absolute entry and direct-script state are not implicated. PowerShell quoting is not the root cause: it supplied the intended venv launcher and the exact application arguments; the venv/base transition changed the process-visible argv[0].

## Filesystem boundary

After the failure, the lower-case R8 family contains only:

- `het_next_l0_ph1_intel_execution_r8_lock.json`;
- `het_next_l0_ph1_intel_execution_r8p1_lock.json`.

No R8P1 result, manifest, commit, independent verification, failure, quarantine, or `.inprogress.*` path exists. The immutable R7D1 failure tree remains one attempt directory containing only its original 931-byte `failure.json`. Thus the attempt produced no candidate evidence and no side effect requiring cleanup.

## Minimal R8P2 evidence revision

R8P2 may correct only the platform invocation model; all runtime, RECORD, RAM, payload, codec, control, stage-hash, transaction and claim contracts must remain identical.

Before execution, R8P2 should freeze two distinct identities:

1. **requested/active venv launcher** — exact `sys.executable`, exact `.venv` `sys.prefix`, launcher SHA, and exact `pyvenv.cfg` SHA;
2. **process-visible base interpreter** — exact `sys._base_executable`, exact `sys.base_prefix`, and the exact `pyvenv.cfg` `home`/`executable` path used at element zero of `GetCommandLineW`, `CommandLineToArgvW`, and `sys.orig_argv`.

Its full expected original vector and raw command-line rendering must use the frozen base-interpreter path at element zero, while its separate launcher/prefix gate must require the frozen venv path. `sys.argv` must remain the exact absolute R8P2 script, `--ack`, and new token with no extras. The independent verifier requires the analogous four-element base-visible command plus the same separate venv-launcher/prefix evidence.

Required negative fixtures should independently mutate: venv `sys.executable`, `sys.prefix`, base executable, `sys.base_prefix`, `pyvenv.cfg` binding, raw command line, parsed argv, `sys.orig_argv`, script, flags, ACK, and extras. Launcher identity and base-process identity must never be conflated into one field again.

Because R8P1 failed before its own writer, R8P2 should bind this immutable diagnosis and retain a bounded, create-new structured failure record if authorization/invocation fails before result publication. This is an evidence/lifecycle repair, not a retry of any scientific or physical arm. A future R8P2 PASS would still authorize no compiler, OpenCL, or device action.

