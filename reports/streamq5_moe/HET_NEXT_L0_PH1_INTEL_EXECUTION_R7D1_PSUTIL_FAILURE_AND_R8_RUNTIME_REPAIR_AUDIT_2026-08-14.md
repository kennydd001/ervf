# PH1 Intel R7D1 psutil failure and R8 runtime-repair audit

Date: 2026-08-14  
Scope: read-only diagnosis and methodology assessment. No payload, OpenCL, or device execution was performed.

## Diagnosis

R7D1 is an immutable **pre-device infrastructure-negative attempt**. It is not evidence for or against the Intel expert mechanism.

The only retained failure bundle contains exactly one 931-byte `failure.json`:

- SHA-256: `88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0`
- kind: `ph1_intel_execution_r7c2_failure` (expected because R7D1 reused the R7C2 outer lifecycle)
- status: `valid_negative_failure`
- stage: `r7a_outer_boundary`
- error: `ModuleNotFoundError:No module named 'psutil'`
- `device_opened=false`
- disposition: `atomic_create_new_bounded_outer_failure`

The traceback reaches `run_het_next_l0_ph1_intel_execution_r7a.py` line 53, the `import psutil` statement. That precedes the payload call on line 56, physical attempt creation/device-open on line 58–60, and all OpenCL allocation/launch work. No R7A output, R7A failure/quarantine, R7A verification, R7D/R7D1 output, quarantine, verification, or in-progress path existed after the attempt. The R7D1 failure directory is now intentionally present and makes the R7D1 authorization non-repeatable.

This record must remain classified as the R7D1 failure. It must not be deleted, moved, relabeled positive, or overwritten by a later result.

## Runtime facts

The failed process used the WindowsApps system interpreter. Both it and the local venv are CPython 3.12.10 on the same base installation, but their package environments differ:

| Property | System interpreter | Local venv |
|---|---|---|
| Python | 3.12.10 | 3.12.10 |
| `psutil` | absent | 7.2.2 |
| NumPy | 2.4.4 | 2.2.6 |

Frozen local-vendor identities:

- `.venv/Scripts/python.exe`: 274,424 bytes, SHA `0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`
- `.venv/pyvenv.cfg`: SHA `9b87fd6636e0e8d878f584a49e365b5e9bdc75507be16f018ee535a69ee1e8fe`
- `psutil/__init__.py`: SHA `7b6a0675824eb1fa2ff0cb1eb36e358dc454703e51dfa4e9a0e6ccd26a159f0c`
- `psutil/_psutil_windows.pyd`: SHA `0035450801bd7d938e9e146c5ec28e619cb5a5f4a18cdc53ac7e9734c7f94f78`
- psutil 7.2.2 `METADATA`: SHA `a263a40220d921d9cb963fc636d34f817aa2eb72c2696e3e3465d088cdb1976b`
- psutil 7.2.2 `RECORD`: SHA `55fd2f55e72c18fd0017a0a033af4661d0227e339c5d772a40a29375e6f740d7`
- NumPy 2.2.6 `__init__.py`: SHA `ad238e76e8c6fbd56a19e6c894864cf466bd2ed76004cac89e78c019fa625607`
- NumPy 2.2.6 `METADATA`: SHA `229f3544b02805e0f6a12030e155d8a45fd3a4100b3291574175e6a76f20e1e1`
- NumPy 2.2.6 `RECORD`: SHA `859c44e1afc26d39b7df8b6b05bee4aed41469d9888c0889710c8603e8520cdc`

An isolated read-only probe with `.venv/Scripts/python.exe -I -B` imported psutil and NumPy successfully with `isolated=1`, `no_user_site=1`, and `dont_write_bytecode=1`. Available RAM at that probe was 49,475,149,824 bytes, above the frozen 16 GiB start gate; this observation is not a substitute for the live R8 resource gate.

## Is a fresh R8 scientifically valid?

**Yes, conditionally.** A new R8 can be a legitimate, separately preregistered runtime-repair attempt because R7D1 failed before payload/device and the scientific source, weights, input, codec, arithmetic, gates, and claim need not change.

It is not valid to call it a simple retry or to claim that only psutil changes: switching to the venv also switches NumPy from 2.4.4 to 2.2.6. R8 must explicitly freeze that environment and prove pre-device preparation equivalence before authorizing one device attempt.

## Required R8 protocol

### Phase R8P — closed no-device runtime/preparation preflight

Run with the exact command shape:

`.venv\Scripts\python.exe -I -B <frozen-r8p-script> --ack <frozen-r8p-ack>`

The preflight must fail closed unless all of the following hold:

1. `sys.executable`, its SHA, Python 3.12.10 identity, venv prefix/base-prefix, `sys.flags.isolated`, `no_user_site`, and `dont_write_bytecode` match the frozen contract.
2. `pyvenv.cfg`, psutil version/files/native extension/distribution record, and NumPy version/files/distribution record match frozen hashes. Prefer validating every installed file against the wheel `RECORD`, excluding only declared cache/bytecode files.
3. `psutil.virtual_memory()` and `psutil.Process().memory_info()` return finite integer telemetry; live available RAM is at least 16 GiB.
4. No candidate runner, backend constructor, OpenCL library, allocation, or launch is reached.
5. Under NumPy 2.2.6, the frozen pure-CPU preparation is replayed: exact 22 controls, record/input/LUT identities, shapes, byte counts, BF16 words, and canonical hashes must equal the existing immutable CPU evidence. No thresholds may be changed and no device output may be observed.
6. The R8P result is atomic, exact-schema, one-shot, and independently verified with negative mutations for wrong interpreter, wrong psutil/native binary, wrong NumPy, missing control, changed preparation digest, non-isolated flags, and insufficient RAM.

This separate R8P is important: otherwise a different NumPy environment is silently introduced at the same time as the psutil repair.

### Phase R8A — fresh one-attempt authorization wrapper

After an independently audited R8P PASS, freeze a new R8 authorization namespace. It must:

1. bind the exact R8P result/verifier and the complete unchanged R7D1→R0 source/evidence chain;
2. bind the exact R7D1 failed-attempt directory, exact one-file set, 931-byte size, failure SHA `88335dc0…`, schema, `device_opened=false`, traceback stage, and disposition;
3. require the R7D1 failure bundle to remain present and unchanged; never call R7D1 `authorize()` because its one-attempt clean gate correctly rejects its now-present failure directory;
4. reconstruct the frozen R7D/PASS9/PASS7/PASS18 authorization, plus the R7A-verifier absence repair, in the new namespace;
5. require R7A output/failure/quarantine/verifier, R7D output/failure/quarantine/verifier, and all new R8 output/failure/quarantine/verifier/temp paths absent, while treating the exact R7D1 failure as required immutable prior evidence;
6. perform the exact interpreter/package checks before importing candidate/physical modules, and preserve `-I -B` in the invocation contract;
7. redirect the audited R7C2 outer lifecycle only to fresh R8 failure/quarantine/revision paths;
8. keep the original resource gates (`start available >=16 GiB`, post-stage available `>=2 GiB`, peak working set `<=12 GiB`), controls, hashes, arithmetic, and numerical gates unchanged;
9. allow exactly one new ACK attempt and forbid automatic fallback to the system interpreter, subprocess retry, cleanup, retuning, or a second device attempt.

The standalone R8 verifier should run under the same exact venv/isolated flags, independently validate both the immutable R7D1 failure lineage and R8 runtime chain, and then replay the frozen numerical/bundle verification.

## Claim boundary

- R7D1 remains a valid negative infrastructure attempt caused by a missing dependency.
- A later R8 PASS would establish only the already-preregistered one-real-expert/input Intel correctness component under the explicitly pinned venv runtime.
- It would not retroactively make R7D1 positive, prove performance, validate a full layer/model, or constitute an industrial breakthrough.
