# PH1 Intel execution R8P2 — Windows launcher/base dual-identity preflight

Date: 2026-08-14  
Status: immutable closed preregistration; no execution authorization.

R8P2 supersedes only the failed R8P1 invocation predicate. R8P1 is an immutable protocol-negative attempt (exit 1 at `exact_invocation`, before runtime, payload, compiler, OpenCL, device, or output). Its diagnosis SHA-256 is `ef42c92407142893532daab1ea5dd7463bec7b384e796fcfe56df59dbbf7a6a7`; its independent source-audit SHA-256 is `85e03d967700d500e3a51d791901110f30ad8be6b9b62723f6f84f9fc610a28e`. No R8P1 result/failure artifact exists, so R8P2 binds the immutable diagnosis rather than inventing one.

The exact launch contract separates two identities:

1. Active venv identity: `sys.executable` is the absolute `.venv\Scripts\python.exe`, `sys.prefix` is the absolute `.venv`, launcher SHA-256 is `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`, and `.venv\pyvenv.cfg` SHA-256 is `9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe`.
2. Base/process identity: `sys._base_executable`, `sys.orig_argv[0]`, and parsed native argv[0] are the exact WindowsApps alias `C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe`; `sys.base_prefix` is `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0`; the real base binary there is 172,912 bytes with SHA-256 `5365b422ee178f691988eb937b7abca5f48910b148f76fcce6dbaf5585c948d0`. The alias is a reparse point and is path-bound, not falsely content-hashed. `pyvenv.cfg` must independently bind its `home` and `executable` to the alias installation.

The preflight native/original vector is exactly `[alias, -I, -B, absolute R8P2 script, --ack, closed token]`; application argv is exactly `[absolute R8P2 script, --ack, closed token]`. Raw Windows quoting is accepted only when `CommandLineToArgvW(GetCommandLineW())` equals that exact vector. Direct entry is mandatory; `-c`, `-m`, reordered flags, extras, wrong token/script, launcher/base/prefix drift, or disagreement among native/original/application vectors is rejected. The independent verifier has its own exact four-element native/original vector `[alias, -I, -B, absolute verifier]` and one-element application vector.

All R8P1 runtime, Python/psutil/NumPy RECORD, 16 GiB start-RAM, CPU preparation, 22 controls, five BF16 stage digests, transaction simulations, topology, and no-device gates remain unchanged. R8P2 adds independent dual-identity mutations and a bounded create-new early-failure artifact for a correct-ACK internal/invocation failure. Invalid argument shape/token returns without filesystem mutation. Failure evidence is capped at 64 KiB, contains no payload, and cannot overwrite a prior attempt.

R8P2 may run only the exact closed no-device preflight after independent source audit. It authorizes no payload, compiler, OpenCL, device, or physical execution.
