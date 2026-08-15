# PH1 Intel execution R8P1 — independent frozen source audit

Date: 2026-08-14  
Scope: read-only/static audit before any R8P1 execution. No candidate import, preflight, CPU payload read, compiler, OpenCL, or device call was performed.

## Verdict

**GO for exactly one closed, no-device R8P1 preflight invocation, and for nothing physical.**

The three blockers from the immutable R8P audit SHA-256 `60518c8999e35e9a09e873c398c4cb18c15a9b5b6e6f0e4972c034f36b3e5a37` are closed. This GO does not authorize the independent verifier, R8A, compilation, OpenCL, or a device attempt. A PASS result must remain a CPU runtime/preparation result only and requires a later independent verification and authorization revision before physical execution.

## Frozen identities

| Artifact | Observed SHA-256 | Status |
|---|---|---|
| R8P1 preflight | `ea51d2399105d038cf273a66014933d9610f8fbc87640526a2617a7d4f29d265` | exact handoff/lock |
| independent verifier | `efc73dead2bd76a24ec4ecac91fc92156821eecccf88ff2149b9f8635b59fd61` | exact handoff/lock |
| preregistration | `9beaaece10b5b7d09234f8e51079cdf582042ccf4ce0d137092511feb63745bd` | exact handoff/lock |
| closed lock | `7fdbde6ceaa354a7766f889b5deaa7d948ce8999d1cf9a7e04754cd343804d60` | closed/PENDING |
| prior R8 independent audit | `60518c8999e35e9a09e873c398c4cb18c15a9b5b6e6f0e4972c034f36b3e5a37` | exact lock binding |

All 16 mutable/transitive chain hashes named by the R8P1 source match the lock. The 13 frozen runtime/preparation fields are retained, so the exact lock has 32 keys. The inherited R8 science/runtime identities are unchanged, including preparation digest `f5a15db125c7a69357574111bd9549c36ae74b67af12205fc71a99a4c8962a49`.

## Closure of the three blockers

### 1. Exact process invocation

The preflight constructs one six-element command vector from the resolved frozen venv interpreter, exact `-I`, exact `-B`, the absolute current script, `--ack`, and the frozen ACK. It independently retains the raw `GetCommandLineW` string, a `CommandLineToArgvW` parse, full `sys.orig_argv`, full `sys.argv`, resolved executable/script paths, absolute-entry evidence, and direct-script state (`preflight`, lines 35–37 and 66–81). Equality is over the complete vectors and exact raw command line; no extra interpreter or application arguments can pass.

Eight mutations cover a `-c` trampoline, wrong script, swapped flags, extra interpreter flag, wrong ACK, extra application argument, altered raw command line, and altered Win32 parse (`preflight`, lines 84–95). The verifier has a separate exact four-element direct invocation contract and repeats the full stored preflight-invocation adjudication (`verifier`, lines 37–41 and 56–75).

### 2. Exact immutable R7D1 failure bundle

The standalone verifier requires one exact attempt directory containing only one exact `failure.json`; its recursive file set must contain that single file. It checks 931 bytes, SHA-256 `88335dc0c7d712d0c2a19a9ee51fe5959f3d725daf2f10d00b8c4a1d9069e3a0`, the exact eight-key schema, covered stages, Boolean `device_opened is False`, disposition, error, kind, stage, status, and traceback markers (`verifier`, lines 78–84).

The current immutable tree satisfies the contract. The TEMP suite has seven outcomes: one positive copied baseline and six required negative fixtures (missing, extra, wrong size/hash, wrong Boolean, wrong disposition, wrong stage), all required true by the verifier (`verifier`, lines 87–99 and 181). The phrase “seven TEMP negatives” should not be used: the frozen preregistration itself enumerates six negatives plus one baseline.

### 3. Transaction and topology evidence

The preflight records the exact pre-run topology before CPU preparation. It requires every R8/R8P1 output, verification, failure, quarantine, and temp path absent and permits only the frozen R8 and R8P1 lock files in the lower-case R8 family (`preflight`, lines 98–109 and 187–195). The current filesystem matches that topology.

Result, manifest, and commit are separately written through exclusive `xb`, flush, `fsync`, create-new hard-link promotion, and temp unlink. The manifest binds result byte count/SHA and the commit binds result and manifest SHAs; partial publication is quarantined (`preflight`, lines 112–151). The seven actual-helper TEMP outcomes cover clean publication, preserved repeat rejection, stale-temp detection/cleanup, failed hard link with no target/temp, post-link cleanup interruption and recovery, repeated verifier rejection, and partial quarantine (`preflight`, lines 154–176).

The independent verifier reimplements the writer and manifest/commit adjudicator, requires the exact committed three-file family and zero unexpected sidecar/temp/failure/quarantine paths, and independently TEMP-tests existing destination, hard-link failure, post-link cleanup interruption, stale cleanup, and repeat-verifier preservation (`verifier`, lines 102–149). It independently validates the complete result schema and rejects six result mutations before publishing its own create-new output (`verifier`, lines 167–183).

## Exact authorized action

Run once, using the exact raw Windows command represented by:

`C:\Users\de_do\Documents\ChatGPT\New project\.venv\Scripts\python.exe -I -B C:\Users\de_do\Documents\ChatGPT\New project\scripts\streamq5_moe\preflight_het_next_l0_ph1_intel_execution_r8p1.py --ack PH1_INTEL_EXECUTION_R8P1_EXACT_FULL_INVOCATION_CPU_PREPARATION_CLOSED`

The executable and script arguments must be quoted by the launcher because their paths contain spaces, and the resulting `GetCommandLineW` text must equal Python's `subprocess.list2cmdline` rendering of the frozen six-element vector. Any mismatch must remain a clean negative, not be worked around in this freeze.

