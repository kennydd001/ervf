# PH1 Intel execution R8A5 — independent frozen-source audit

Date: 2026-08-14  
Scope: read-only audit of the exact frozen R8A5 runner, standalone verifier, preregistration, lock, explicit module resolution, authorization order, local lifecycle, terminal adjudication, and fresh topology. No runner/verifier execution, payload read, compiler, OpenCL, or device call was made.

## Verdict

**GO for exactly one physical R8A5 attempt under the frozen `.venv -I -B` command and ACK, followed by the frozen standalone verifier.**

R8A5 closes the deterministic R8A4 module-depth failure. Authorization and execution now use explicit, exact-hash module bindings rather than revision ancestry. The module-resolution sentinel and inherited R7A authorization are read-only and occur before physical remapping, payload construction, OpenCL, or device access. Correct direct invocations that fail resolution/authorization now retain bounded infrastructure evidence. The frozen R7A scientific route and the independently audited R8A4 terminal adjudication remain unchanged.

This is source-level execution authorization, not a scientific PASS. Only the immutable physical artifact plus the standalone verifier can establish the component result.

## Frozen package verified

- runner: `scripts/streamq5_moe/run_het_next_l0_ph1_intel_execution_r8a5.py`, SHA-256 `1422fe70e2b0c33f19c1df969a40f7a7414b8a3734cc9914e3f687a5fcc25168`, 16,708 bytes;
- standalone verifier: `scripts/streamq5_moe/verify_het_next_l0_ph1_intel_execution_r8a5.py`, SHA-256 `75168d7502a141291f3b7459f779ae92439b7ffd32df667875fd73a365e62a66`, 26,290 bytes;
- preregistration: `reports/streamq5_moe/HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_PREREGISTRATION_2026-08-14.md`, SHA-256 `b7788e4185b29c8a6f194d0dbf96fc8a9e6b9bed78eba87e49d82b8d70c4b056`, 6,054 bytes;
- open lock: `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8a5_lock.json`, SHA-256 `13be47460512fe42a0a4dbe2995c2299e5bf02f75db85058b21296047cbe7979`, 1,232 bytes;
- bound R8A4 GO audit: SHA-256 `4dc0c0a1f3e411f78f81ef667baf6e00e94f73204f0ed9b3d1794b0f300a7438`;
- bound R8A4 failure diagnosis: SHA-256 `d88ff5fd76e11757d7d53acf7279bc52897b9b74169414daee3ce18cd8bc6b21`;
- exact token: `PH1_INTEL_EXECUTION_R8A5_AFTER_R8P8_PASS_AND_EXPLICIT_BINDING_AUDIT_GO`;
- all six current output/failure/quarantine/verifier paths and every matching R8A5 in-progress path were absent at audit time.

## Explicit module-resolution repair

The runner imports exactly:

- `run_het_next_l0_ph1_intel_execution_r8a` as the historical authorization contract, frozen SHA-256 `552a7f08f83f2ba2ce3da29581029dfdd79e86fbb75faeb71356965073228f15`;
- `run_het_next_l0_ph1_intel_execution_r7a` as the physical implementation, frozen SHA-256 `01fa21266137335494de2d21adba11f45fe83ff95f660d90cef7acc389c1cb04`;
- the physical numerical verifier by path, frozen SHA-256 `18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88`.

There is no `prior.prior`, `ancestor`, `frozen` ancestry traversal, R8A1–R8A4 wrapper import, or candidate-runner import in the standalone verifier. The runner AST gate enforces the explicit import set and rejects the forbidden wrapper imports and multi-level ancestry aliases.

`resolution_sentinel()` checks the actual module `__file__`, exact SHA, and required attributes. It evaluates only immutable historical predicates: R8P8, R7D, and the exact R7D1/R8P6 failures. Importing the frozen modules defines constants/classes and imports NumPy-backed CPU helpers but does not read the selected shard/D2 payload, construct the package, load an OpenCL library, instantiate the backend, or open a device. `physical.authorize()` then verifies only frozen locks, manifests, hashes, and prior PASS artifacts.

## Authorization and execution order

For the exact direct command, the runner order is:

1. exact Python argv;
2. native/Python `.venv\Scripts\python.exe -I -B` identity and direct-entry validation;
3. clean current and R8A–R8A4 terminal topology;
4. AST gate;
5. explicit module-resolution sentinel and historical predicates;
6. exact 12-binding current lock, R8A4 audit, and R8A4 diagnosis;
7. frozen read-only R7A authorization;
8. fresh R8A5 physical path remapping;
9. one call to frozen `physical.execute_authorized(auth)`.

The direct R7A module's `execute_authorized()` invokes its own `configure()` after the R8A5 remapping, so the shared atomic bundle/failure helpers receive the fresh R8A5 output, backend-failure, and backend-quarantine roots. Payload construction, RAM gates, OpenCL, device work, ledgers, outputs, cleanup, and resource sampling remain byte-identical frozen R7A logic.

Wrong argv/ACK, non-direct entry, or nonclean/already-used topology returns nonzero without mutation, as preregistered. After a correct direct invocation and clean gate, a resolution/authorization exception is caught and written as exactly one bounded `predevice_resolution_or_authorization` failure with `device_opened=false`; it is terminal-invalid and cannot become a scientific negative.

## Local failure and terminal lifecycle

- A complete committed R7A bundle is authoritative and returns according to its positive flag.
- A delegated nonzero without commit is correlated only to one newly created, single-file, bounded R7A failure with valid schema and `device_opened=true`.
- Missing, multiple, extra, mismatched, predevice, bare-nonzero, or zero-without-commit evidence becomes invalid protocol.
- An unexpected delegated exception without commit is retained as bounded early infrastructure evidence and remains terminal-invalid.
- All current writers are create-new, fsync their content, use a fresh nonce, and never overwrite an existing attempt.
- A current or ancestor terminal artifact prevents authorization, enforcing the one-attempt/no-retry rule.

The standalone verifier independently redefines the current invocation, chain, extension, exact committed bundle, recursive failure trees, topology, correlation, and writer. It does not import R8A5. It retains the audited 31-case production terminal matrix with injected roots and the frozen R8A2 committed allowlist: only `stages` and/or `counters` may be false in an allowed committed device negative; all protected gates must remain true.

## Claim boundary

- Only `positive` returns verifier exit code zero.
- Allowed committed or correlated post-device negatives remain structured negative evidence and do not pass.
- Early infrastructure evidence is invalid for the scientific mechanism.
- Scope is one official real expert/input Intel correctness component only.
- No throughput, full-layer, full-model, heterogeneous, industrial, or breakthrough claim follows from this source GO.

## Authorized sequence

Exactly one physical command is authorized:

```powershell
& '.\.venv\Scripts\python.exe' -I -B 'C:\Users\de_do\Documents\ChatGPT\New project\scripts\streamq5_moe\run_het_next_l0_ph1_intel_execution_r8a5.py' --ack PH1_INTEL_EXECUTION_R8A5_AFTER_R8P8_PASS_AND_EXPLICIT_BINDING_AUDIT_GO
```

Do not retry, retune, or reuse R8A4. After this one attempt, run the exact frozen R8A5 standalone verifier under the same `.venv -I -B` runtime and adjudicate only the immutable evidence. This audit executed neither command.
