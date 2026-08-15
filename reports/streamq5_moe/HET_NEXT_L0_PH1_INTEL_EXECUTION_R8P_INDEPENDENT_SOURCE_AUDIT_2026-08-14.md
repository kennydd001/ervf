# PH1 Intel execution R8P — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit before execution. No candidate import, preflight, payload read, compiler, OpenCL, or device call was executed.

## Verdict

**NO-GO for the exact R8P no-device command in this freeze.**

The runtime, wheel, codec, controls, stage hashes, and preparation digest are internally consistent. Two provenance/independence gaps and one transaction-adjudication gap remain. These do not require scientific or numerical changes; they require a fresh immutable R8P1 source revision.

## Frozen identities and independently checked invariants

| Artifact | SHA-256 | Status |
|---|---|---|
| R8 runner | `836ec10eeb5a8af58f9eb108d5c7c0acfaa350cb41d862361b6775008553f10f` | exact handoff/lock |
| closed R8P preflight | `9e1aa03b8afa341570f5a735dc1777441050074d75365daa605150efb85c4522` | exact handoff/lock |
| R8P independent verifier | `a96ec298cd9d8bb3a3e0eb043dc9e4db3c258b2e284550ee8cb3086003e0f869` | exact handoff/lock |
| future physical verifier | `afc162c7238791c888e73903c5c2e97149a098872cf7ed613bf68a1903406b6d` | exact handoff/lock |
| preregistration | `99ec4c359f17c30f01c5d2616be840bf00dbff4a7e58c5d4239118efb631f5d8` | exact handoff/lock |
| closed lock | `0be70e2cccc5adafcb091464af9109cb8efa2f5cca553eb60a63f64b5268203a` | closed/PENDING |

Independent read-only checks established:

- all 33 chain artifacts hash exactly to the lock;
- psutil RECORD: 28 rows = 17 verified hashed files + 10 excluded bytecode/cache rows + the sole unhashed RECORD row; zero mismatches;
- NumPy RECORD: 1,311 rows = 899 verified hashed files + 411 excluded bytecode/cache rows + the sole unhashed RECORD row; zero mismatches;
- the hardcoded 22-control preparation summary independently canonicalizes to `f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49`;
- the three 675,840-byte records, 4,096-byte input, 131,072-byte LUT, and five frozen stage hashes/shapes agree across production-builder, frozen verifier, and immutable CPU evidence contracts;
- R8 imports R7D dynamically only after runtime, failure, chain, lock, and clean-state gates; it does not import or call R7D1 authorization;
- the R7D1 failure file is currently the only file in its failure tree and has exact size 931 and SHA `88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0`;
- every R8 result/output/failure/quarantine/verifier path is absent and the R8 in-progress count is zero.

## Blocking findings

### 1. Exact invocation is not self-bound

The preregistration authorizes one exact command with `.venv\Scripts\python.exe -I -B`, the exact preflight script, and the exact closed ACK. The runtime evidence retains only `sys.orig_argv[1:3] == ["-I", "-B"]`.

It does not retain or validate:

- the original interpreter target after normalization;
- the exact script path at `sys.orig_argv[3]`;
- the complete original argument vector and absence of extra interpreter options;
- exact `sys.argv` script/`--ack`/token structure.

Consequently a `-c`/import trampoline or additional interpreter flags can satisfy the recorded runtime contract even though it is not the frozen invocation. The ACK parser alone does not prove which source file was the original entrypoint. The independent verifier repeats the same partial first-two-flags check and has no separately frozen invocation contract.

Required repair: retain and validate a normalized invocation record with exact executable, `-I`, `-B`, exact current script path, exact argument names/order/token, and no extras. R8P1 and its verifier require separate expected script/ACK contracts. Add negative cases for `-c`, wrong script, swapped flags, an extra interpreter flag, wrong ACK, and an extra application argument.

### 2. The R8P verifier does not independently validate the immutable R7D1 bundle

The preflight correctly calls `runner.prior_failure_valid()`, which checks the exact one-file tree, size, hash, and core schema. The purported independent R8P verifier does not implement or call that bundle validator. Its `valid_result()` only compares the stored failure SHA with the current `failure.json` SHA.

An extra file added to the R7D1 failure directory after R8P, or a directory-cardinality violation, is therefore invisible to the independent verifier. It also does not independently assert the exact kind/status/stage/disposition/covered-stage schema. A file hash is sufficient for that file's bytes, but not for the required exact bundle topology.

Required repair: implement a standalone R8P-verifier bundle check over the exact failure root: one exact file, 931 bytes, exact SHA, strict JSON key/schema/value contract, no extra files/directories relevant to the bundle. Add real TEMP negative fixtures for missing, extra, wrong-size/hash, wrong Boolean type, wrong disposition, and wrong stage.

### 3. Independent transaction cleanliness is incomplete

Both writers use a useful create-new sequence: exclusive temporary creation, flush/fsync, hard-link promotion, and temporary unlink. The preflight also records result/verifier/temp absence before writing.

However the independent verifier does not require the R8P result temporary glob to be empty, does not independently recompute the full R8 clean-state/failure topology, and does not exercise interrupted promotion or cleanup failure. Its `result_sha256` check is tautological (`hash(raw bytes) == hash(the same file)`) rather than a transaction/topology check.

This permits a promoted PASS result plus a leftover `.inprogress.*` link to be independently accepted after a cleanup interruption. It also leaves the create-new/failed-promotion behavior untested despite the requested lifecycle/transaction audit.

Required repair: before verification, require exact result path/type/size plus no result/verifier temporary glob and no unexpected R8 sidecar/output/failure/quarantine path. Add TEMP tests for existing destination, stale temp, hard-link failure, post-link cleanup failure, and repeated verifier execution. The verifier should bind the independently computed result SHA in its output, but should not count hashing one file twice as a separate gate.

## Non-blocking boundary for future R8A

The current R8 runner and future physical verifier intentionally do not consume R8P PASS artifacts because R8A has not been authored or opened. That is consistent with the preregistration, but these files cannot be authorized physically as-is. A later R8A must bind the exact R8P result plus independent-verification result, preserve the R7D1 negative bundle, and receive another audit.

No arithmetic threshold, source slice, input, LUT, control, stage hash, resource gate, device backend, or claim needs to change to repair R8P.
