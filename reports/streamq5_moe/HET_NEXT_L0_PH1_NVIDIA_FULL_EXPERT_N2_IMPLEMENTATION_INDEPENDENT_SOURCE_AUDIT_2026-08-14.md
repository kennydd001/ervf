# PH1 NVIDIA full-expert N2 implementation — independent source audit

Date: 2026-08-14  
Mode: source-only. No candidate import, static preflight, payload read, compiler load, CUDA load or device call was performed.

## Verdict

**NO-GO for the frozen N2 no-device static preflight.**

N2 closes the broad terminal-classification error and materially improves control evidence and physical-ledger checking, but it does not close all seven N1 source-audit blockers. The remaining gaps are provenance, non-vacuous static validation, exact ABI/runtime evidence and failure-path truthfulness.

## Frozen integrity and absence

All handed-off hashes match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| common | 20,056 | `7abefeda869a2ba3c4e64778a27d391640bdd75cfb6850fe6da435a61d9e99e7` |
| CUDA kernels | 6,167 | `7326fedd46459671ead70ade7458b75ef5617354c8f8c454b7842bc62838b555` |
| backend | 30,712 | `a11976af71068dc96672f738363c5fa66c9520a151f62a2495bd6e227a0c2131` |
| transaction | 6,434 | `766dd6069b8551dcdfb1bcd0bd9fc1e0e72ec65fc15708af264bcd38b09d3095` |
| runner | 14,166 | `819b1d62af53e9fd0b3398a67c683fdf95d56dce2811f06194a7ab1e10bf9f54` |
| verifier | 26,374 | `c60387b04ee448d25d9dac938e824b2352012bef0805cc4cc5375cbde366d6ca` |
| static preflight | 12,498 | `27ee77bcbe98bad2b80b63db1f6edefe855e97f438e40dd25aaa11b14b0149e3` |
| source preregistration | 2,410 | `540497e630f3eb049a6fe6fcb94d1bc374a321ec5472c70a037d947b82b10e22` |
| verifier lock | 2,268 | `2f50072a42158a949d9c4bc205cd477658965e85676fb1b996ddeae950eae0ab` |
| source lock | 4,727 | `0af6a236710fcf323e630db1c81e2d235f0966d396234370974ea81f7723c775` |
| preflight lock | 2,221 | `ca01acc449b4788292e0360ef6f9d4f744036f5830e7898bcde426d5ff989239` |

Every present lock entry rehashes true: verifier 11/11, source 21/21 and preflight 11/11. Compile output, physical output, compile/physical failure roots, quarantine, static-preflight result and independent-verification result are absent.

## Blocking findings

### 1. Verifier and preflight provenance is incomplete and partly circular

The verifier lock has only 11 bindings. It does not bind the N2 source lock, preflight source/lock, N1 design audit, R1/R2 contracts and audits, CPU package, or R8A5 bundle. Yet `verify_het_next_l0_ph1_nvidia_n2.py:21,156-163` reads `SOURCE_LOCK` and trusts its `cuda_source` entry to authenticate candidate `source.cu`. Since `SOURCE_LOCK` itself is not verifier-lock-bound, that proof is mutable outside the verifier’s provenance closure.

The preflight lock likewise has only 11 bindings and omits the direct N1 design/R1/R2/Intel chain. This conflicts with the frozen N1 requirement that later locks and independent verifiers bind those artifacts directly, not indirectly. `preflight...:31-35` verifies only entries that happen to be present, so both reduced locks pass their current hash gates.

### 2. The static ABI/schedule/kernel gate is still non-vacuous only in name

`preflight_het_next_l0_ph1_nvidia_n2_static.py:44-61` checks the 30 ABI **names**, sees one generic `.argtypes` assignment and one `.restype` assignment, and looks for a few ctypes names. It never compares the 30 exact argument vectors or return types. Changing an individual signature, pointer depth, `_v2` operand, module option, stream flag or launch operand survives `abi_contract`.

`schedule_contract` checks only that buffer-name constants and four call names occur. It does not prove 9/5/4/9/1/7 cardinalities, ordering, stream/pointer operands, context order, seven exact sample slots, or absence of Driver calls after release. `kernel_contract` remains the N1 substring checker; row mapping, record offsets, loop bounds, reductions and integer BF16 arithmetic can change without rejection.

No source mutation suite exercises these ABI/schedule gaps. Therefore the N2 preregistration claim that static preflight AST-audits all exact signatures and schedule fields is false.

### 3. Static preflight imports candidate modules and mutates only a toy verifier snapshot

`preflight...:83-105` executes the candidate verifier and backend with `spec.loader.exec_module`. This contradicts the inherited text/AST-only, no-candidate-source-import boundary. The backend import also executes its top-level `psutil` and candidate-common imports.

`verifier_mutations` calls only `contract_snapshot_valid` on a small hand-built dictionary. It never constructs or mutates a compile/physical bundle and never calls production `verify_compile` or `verify_physical`. Consequently the actual loader/ABI/pointer/resource/control/compile parsers receive no negative test. `cleanup_faults` covers only the 30 ordinary releases; it does not test context-pop/restore/primary-release failures, acquisition failures, meminfo placement, post-release bans or full failure evidence.

### 4. Exact ABI and physical ownership evidence remains incomplete

`verify_het_next_l0_ph1_nvidia_n2.py:181-193` is stronger than N1, but its ABI gate checks the function-name set, a common `c_int` restype, and only a substring in the `cuMemAlloc_v2` first argument. It does not compare the exact 30 frozen argument vectors. It also omits exact validation of:

- stream-create flag/registered ownership;
- module-load options/pointers and exact two function handles;
- context retain registration, push operand and owner thread consistency;
- pinned-write payload hashes/pointers;
- owner-thread equality across ordinary/context rows.

The runtime-module scan (`backend:213-225`) filters on tokens in the **basename** before it checks a full path for `cupy`; a CuPy `.pyd` under a `cupy` directory whose basename lacks that token is skipped. The scan occurs immediately after NVCUDA load, not again after execution, so later-loaded forbidden modules are not observed. The verifier therefore still cannot establish the full frozen forbidden-runtime claim.

### 5. Post-device protocol failures lose device evidence and are falsely marked device-closed

After `DriverBackend.run` succeeds, runner protocol gates may fail at `run_het_next_l0_ph1_nvidia_n2.py:126-148`, or independent precommit verification may fail at line 153. Those exceptions are ordinary `RuntimeError`, not `DriverFailure`. The catch at lines 154-156 therefore sets `partial={}`, writes `device_opened=false`, and discards the complete successful backend evidence, resources and cleanup state even though the Driver was opened and kernels may have run.

This is a truthfulness and lifecycle blocker. A failure after device execution must retain the already available backend evidence and report `device_opened=true`; it must remain infrastructure-invalid, not look predevice.

### 6. Authorization/failure semantics regress the frozen mutation-free boundary

The exact ACK is checked before the outer `try`, but a correct ACK against a closed/drifted lock enters the catch at `run...:167-177` and creates a failure artifact. Thus invalid authorization can mutate the failure namespace, contrary to the inherited rule that invalid authorization writes nothing.

Additionally, the backend imports `psutil` at module top (`backend:15`) before runner authorization. The lock does not bind an interpreter/environment contract. In a Python environment without psutil, the runner fails during import before `main` and cannot create bounded evidence.

### 7. Compiler-failure and transaction tests remain incomplete

The explicit intended NVRTC suffix at `backend:86-149` is an improvement for returned error codes. A ctypes exception occurring during a call before `row(...)` is recorded, however, is incorrectly converted into `attempted=false`; a ctypes exception during destroy produces no destroy row. No static fixture exercises null/non-null create failures, compile failure plus log, every retrieval failure, host exception or destroy failure.

The transaction suite remains the N1 minimal fixture (`preflight:72-82`): it does not test prelink/postlink publish failures, verifier rejection, oversize evidence, failure-writer failure, runner recovery branches or primary/secondary error preservation. Static-preflight and standalone-verifier outputs are still direct non-fsynced `write_text`/`open("xb")` writes without bounded failure handling.

## Repairs that are genuinely present

- Full requested/presented digests, headers and checker-stage traces are retained for all 22 predevice controls; the verifier independently reconstructs them, including wrong LUT.
- Candidate `source.cu` is checked against the frozen CUDA SHA in the compile verifier.
- Seven meminfo rows, pointer/stream crosslinks, return codes, 30 releases and exact raw outputs/counters receive materially stronger physical verification.
- Runner protocol gates are separated from the two numerical gates before a terminal numerical result is formed.
- The CUDA arithmetic source is unchanged from N1 except a trailing newline; no new static arithmetic contradiction was found.

These improvements do not offset the blockers above.

## Required next revision

Before any no-device static preflight, a fresh immutable revision must:

1. restore the complete direct lock/provenance closure, including binding the source lock read by the verifier;
2. compare exact ABI AST structures and exact schedule/control-flow relations, with one production-relevant mutation per field;
3. test actual `verify_compile`/`verify_physical` paths on bounded synthetic bundles without candidate imports or payload reads;
4. complete ABI/ownership/owner-thread/module/function/runtime-module validation;
5. preserve backend evidence and true `device_opened` state for every post-device failure;
6. keep invalid authorization mutation-free and bind the exact runtime dependency environment;
7. add full compiler and transaction fault matrices with exact attempted/not-attempted and primary/secondary failure evidence.

Compile and physical execution remain closed.
