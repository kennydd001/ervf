# PH1 Intel execution R7C2 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit; no candidate import, preflight, payload, compiler, OpenCL, or device call was executed.

## Verdict

**GO for exactly one execution of the currently closed R7C2 no-device static preflight.**

This is not authorization for payload access or a physical Intel run. R7C2 remains closed (`execution_open=false`, `audit_token=PENDING`). Physical authorization requires a separately frozen revision after a clean passing preflight and another audit.

## Frozen identities

| Artifact | SHA-256 | Status |
|---|---|---|
| runner | `7eb79d9c9be2ce682c4d52a2df5c3f53e55528fde0afbcb1a23d824829a8b2b6` | matches handoff and lock |
| independent verifier | `0fb1fa70e709b41ee188837ed288789709ea3763e2fb6198378d8384bb0d3711` | matches handoff and lock |
| closed preflight | `e555a18a138bb63b3b39f4e45ae9793e83b2eface62612c41f8e3d409081150d` | matches handoff and lock |
| preregistration | `a4caaa7b0e3644d58a07bdb881d2cc0a94df498c769a125bbc9f050924a7655d` | matches handoff and lock |
| closed lock | `ad42f40137078d80af858a60d73acec1b999b4418455cfa47ff25cf6a03d2456` | matches handoff |
| R7C1 independent audit | `e75ae1897d4ce73664c1225ca499e41a029660f0506b2fc72ee4cc65ddfadeb2` | directly bound |

The current runner, verifier, preflight, preregistration, R7C1 audit, immutable R7A authorization result, and R7P result all hash to the values bound by the closed lock. The R7A physical output, R7A failure/quarantine trees, R7C2 revision output, R7C2 failure/quarantine trees, R7C2 preflight result, and R7C2 verifier result were absent during this audit.

## Closure of the two R7C1 blockers

### 1. Exact inherited device-state retention — closed

- `inherited_evidence()` retains `payload["device_opened"]` only when its JSON type is exactly Boolean.
- A valid inherited R7A failure requires that exact Boolean plus the frozen kind, status, error, and disposition contract.
- Malformed, unparseable, oversized, over-cardinality, and inspection-error evidence records `inherited_device_opened=null` and cannot satisfy the exactly-one-valid gate.
- The delegated top-level `device_opened` is now computed solely as the OR of retained exact Boolean values. It is no longer inferred from disposition.
- The preflight adds three direct cases: valid `true`, valid `false`, and wrong-type string `"true"`. It requires correct retention, top-level propagation, and rejection of the wrong type.

The prior nine lifecycle cases remain exercised through the frozen R7C1 fixture: early exception, structured return 3, bare return 3, multiple failures, oversized failure, stale temporary state, valid positive commit, valid negative commit, and success without a commit.

### 2. Complete material clean-state gate — closed

The preflight requires exact immutable presence/hashes for the R7A authorization result and R7P result. It separately requires absence of:

- R7A physical output, failure tree, and quarantine tree;
- R7C2 revision output, failure tree, and quarantine tree;
- the current R7C2 preflight and independent-verification result files;
- R7A output/failure and R7C2 output/failure `*.inprogress` paths;
- every report-root `*.inprogress` path whose name contains `r7a` or `r7c2`.

The clean-state evidence is retained in the preflight result, so a non-clean state fails rather than being silently repaired.

## Regression and verifier boundary

- R7C2 delegates the physical computation to the frozen R7A runner. It changes no payload, OpenCL backend, source/Q5 arithmetic, numerical gate, resource gate, or scientific claim.
- Authorization is still fail-closed: the closed lock cannot authorize physical execution; the future open lock must bind every current/transitive artifact and the exact token.
- The independent verifier does not import R7C2, R7C1, R7C, or R7B candidate runners. It independently validates the R7C2 authorization extension and only then loads the hash-pinned frozen R7A numerical verifier.
- Eight authorization-extension mutations remain required to fail.
- Claim boundary remains: one real expert/input Intel correctness component only.

## Authorized next action

Run the exact frozen no-device preflight once. Accept it only if every named check passes, the retained evidence reports the expected true/false/wrong-type device cases and all nine inherited lifecycle cases, the clean-state record is complete, and the produced result is hash-bound in a new immutable authorization revision. Do not proceed directly from this audit to payload or OpenCL execution.
