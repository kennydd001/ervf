# PH1 Intel execution R7D — independent final physical source audit

Date: 2026-08-14  
Scope: static/read-only audit. No candidate import, payload read, compiler, OpenCL, or device call was executed.

## Verdict

**NO-GO for the physical ACK run.**

The authorization chain and delegated lifecycle are otherwise internally consistent, but the live clean-state implementation is weaker than the frozen preregistration. One small immutable repair is required before physical execution.

## Frozen identities and live state

| Artifact | SHA-256 | Status |
|---|---|---|
| R7D runner | `e67535b80cdee72c9de24a64842447b0639bf783789329e345b0b4461da621f9` | matches handoff/open lock |
| standalone verifier | `8fa44558412eed80891d013fda8a08881e65ca30caf35062c9b7428a02d10fb4` | matches handoff/open lock |
| preregistration | `9b23886235aa7556f14fc793c5a6ac53e27b77343732c2fc6568f59d3363f3c4` | matches handoff/open lock |
| open lock | `fa2a514af78ac75cd94376f8d04d801fd1dfa27592b595bf42a662bab3e15658` | exact token/open state |
| R7C2 PASS 9/9 | `de8745c02cd0b2951adbb04338cf350704608023530edd91a260b73880ebcd8c` | exact hash and schema observed |
| R7C2 audit | `d6f9ca23a43bef30c0a907efa7997f2d755e6fa3a32c0ed5dab1c21498863a5e` | directly bound |
| R7A PASS 7/7 | `a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265` | exact hash and schema observed |
| R7P PASS 18/18 | `e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959` | full sentinel/mutation evidence observed |

All 43 file hashes represented in the R7D chain matched the open lock. At audit time, the R7A physical output/failure/quarantine/verifier result, R7D output/failure/quarantine/verifier result, and matching R7A/R7D in-progress paths were absent.

## Blocking finding

### Live authorization omits the R7A verifier-result path

The preregistration requires current absence of all R7A/R7D output, failure, quarantine, **verification**, and matching in-progress paths before any filesystem mutation or payload/device activity.

`clean_now()` in the R7D runner checks:

- `physical.OUT`, `physical.FAILED`, and `physical.QUAR`;
- the R7D revision output, failure, quarantine, and R7D verifier result;
- every root-level R7A/R7D `*.inprogress` path.

It does **not** check the distinct frozen R7A verifier-result path:

`reports/streamq5_moe/het_next_l0_ph1_intel_execution_r7a_independent_verification.json`

That file is currently absent, but absence observed during this audit is not a substitute for the promised live fail-closed check immediately before physical execution. If the file appears after this audit, authorization still succeeds, contrary to the immutable preregistration.

## Exact required repair

Create a fresh immutable authorization-only revision that:

1. defines the exact R7A verifier-result path;
2. includes it in the `clean_now()` absent tuple before `physical.authorize()` and before lifecycle/path mutation;
3. preserves the current R7C2/R7A/R7P parsers, full chain, exact ACK discipline, physical computation, R7C2 lifecycle, verifier, and claim unchanged;
4. remains one-attempt fail-closed through R7A/R7D output/failure/quarantine/temp state.

A tiny static/source check should prove that both the R7A and new-revision verifier-result paths participate in the live authorization predicate. After refreeze and independent confirmation, exactly one ACK run may be authorized, followed by the standalone verifier regardless of whether the physical result is positive or negative.

## Checks that passed

- Exact R7C2 9/9 schema, nine lifecycle cases, three device-state cases, eight rejected extension mutations, and retained clean-state evidence are consumed.
- Exact R7A 7/7 and R7P 18/18 contracts are consumed before payload/OpenCL.
- The open lock and exact ACK bind the complete R7C2-to-R0 chain.
- No filesystem path is reassigned before authorization completes.
- After authorization, R7C2 failure and quarantine semantics are redirected to R7D paths while the physical output remains the frozen R7A output.
- The standalone verifier imports no R7B/R7C/R7C1/R7C2/R7D candidate runner. It reconstructs the authorization extension first and only then imports the hash-pinned R7A numerical verifier.
- Claim remains limited to one real expert/input Intel correctness component.
