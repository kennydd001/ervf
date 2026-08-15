# PH1 Intel execution R7A — independent frozen authorization audit

Date: 2026-08-14  
Scope: static/read-only audit. No authorization preflight, payload, compiler, OpenCL, or device call was executed.

## Verdict

**GO for exactly one execution of the frozen no-device R7A authorization preflight; NO-GO for the physical R7A runner under the current freeze.**

The authorization preflight is sound and device-free. The physical runner, however, does not consume or validate the authorization-preflight result. Because its lock is already open and the ACK is already frozen, the physical runner can be invoked directly while the authorization-result file is still absent. That is a real bypass of the requested `auth-preflight → physical run → verifier` sequence.

## Frozen identities and current state

Observed SHA-256 values match the handoff:

- runner: `01fa21266137335494de2d21adba11f45fe83ff95f660d90cef7acc389c1cb04`
- verifier: `18b64765469e38c5211d28afe586e0a559e97f6e2110f09f54c4f58d9c38dd88`
- authorization preflight: `46fd3f180e54f8b2367615d823f597d00e408b6ee39ba698c56c1a973e0828cc`
- preregistration: `b051870a889f23c22d25c2ed12fa14d9283f42e1bbebe9f3734bbd6210f1596d`
- open lock: `87ca1118b6656af25a5353287edfdd515e5d97dea08862fec805986ecd1cf3d2`
- R7P PASS result: `e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959`
- backend/common remain `8bbfa1a6...` / `d6abe579...`

The R7A output directory, R7A authorization-preflight result, and R7A independent-verification result are absent. The exact open token is `PH1_INTEL_EXECUTION_R7A_AFTER_R7P_PASS18_AND_FINAL_AUDIT_GO`.

## Checks that pass

1. **R7P evidence.** The immutable result is genuinely PASS 18/18 with `no_payload_compiler_device=true`; all 18 named top-level checks are true; the 20-entry verifier baseline is all true with an empty false list; the exact 28 mutation names equal the exact 28 rejected names; both corrected all-row sentinels have expected/repeated full digests; and both poisoned write-after-loop negative shapes pass their deterministic evidence contract.

2. **Runner scope.** Direct R7-to-R7A diff is authorization/namespace-only: R7A paths, bundle/failure kinds, token, and R7P bindings. Numerical gates, resource limits, expected stage hashes, buffer/launch handling, controls, and device invocation are unchanged. The backend/common sources remain byte-identical.

3. **Verifier scope.** Direct R7-to-R7A diff is limited to paths, R7A kinds, and R7P provenance. The fixed row assignment remains inside the loop. Codec, integer FMA, BF16 operations, reduction, controls, oracle, and all verification predicates are unchanged. The authorization preflight also AST-compares the exact numerical verifier functions.

4. **Authorization ordering inside the runner.** Given an invocation, `execute()` parses the exact ACK and calls `authorize()` before `execute_authorized()`. Filesystem recovery/configuration, RAM sampling, payload loading, backend construction, OpenCL loading, allocation, and execution all occur only later. Wrong ACK and authorization failure return without filesystem writes. Importing the backend module does not construct `Backend` or load OpenCL; that occurs only in the later backend run path.

5. **Authorization preflight safety.** It reads/hashes/parses sources and frozen JSON only, performs AST comparisons, checks output absence, and writes only its result. It neither imports the runner/backend nor invokes payload/compiler/device APIs. Its seven gates are non-vacuous for the frozen evidence.

## Blocking bypass

The R7A runner's `authorize()` validates the open lock, exact token, its source/backend/common/verifier/preflight/prereg hashes, compile/CPU packages, and the R7P artifact hash. It does **not** define an authorization-result path, read `het_next_l0_ph1_intel_execution_r7a_authorization_preflight.json`, validate its kind/check map/PASS count, or bind its SHA-256.

The open lock likewise contains `authorization_preflight_result_absent=true`, not a PASS-result hash. Therefore the current state already satisfies the physical runner's authorization predicate. Supplying the published ACK bypasses the still-unexecuted authorization preflight.

External operator discipline is not an executable no-bypass gate. The requested ordered chain is not enforced by the frozen runner.

## Minimal next immutable step

1. Execute only the exact R7A no-device authorization preflight once.
2. Accept it only if all seven gates are true, `passed=total=7`, `no_payload_compiler_device=true`, the exact ACK is recorded, and the R7P result hash is `e10c513f...`.
3. Freeze a fresh authorization-only R7B (or R7A1) runner/lock revision. Before any filesystem recovery, payload read, or backend construction, its `authorize()` must:
   - require the exact authorization-preflight result file;
   - bind its SHA-256 in the new open lock;
   - parse and require the exact kind, seven expected check names all true, `pass=true`, `passed=total=7`, `no_payload_compiler_device=true`, exact ACK, and exact R7P result hash;
   - bind this audit and the complete unchanged R7A/R7P chain.
4. Keep backend/common, numerical code, verifier arithmetic, stage hashes, controls, resources, buffers, launches, and claim unchanged.
5. Independently audit that new authorization-only revision. Only then may one physical attempt be authorized, followed by the exact independent verifier.

## Claim boundary

No physical Intel result exists. This audit authorizes only a device-free authorization preflight and does not establish PH1 correctness, performance, or a model-level result.
