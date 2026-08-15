# PORT80B-D10B-R second-audit count correction

Date: 2026-08-13

This correction supersedes
`PORT80B_D10BR_SECOND_AUDIT_REPORT_ERRATUM_2026-08-13.md`.

The immutable second independent-verification JSON (SHA-256
`e2133ab238746548798336f5d445a2ed1f693131ddbfcb47e2791d3c6be9ea61`)
contains **49** named true entries in `checks`, `check_count=49`,
`checks_passed=49`, and 19/19 true entries in `recomputed_gates`.

The 49th check is legitimate: it verifies that the active bulk bank exists
with the exact frozen size of 49,925,652,480 bytes. The audit also validates
the manifest's declared bank SHA-256, but deliberately does not rescan the
49.9-GB payload. Therefore the correct reading is **49/49 independent checks
and 19/19 recomputed frozen gates**.

The earlier erratum incorrectly stated that the JSON contained 48/48 and that
the generated report's 49/49 was an off-by-one. Neither statement is correct.
No raw result, independent-verifier JSON, generated report, runner,
preregistration, preflight, manifest, or bulk bank was modified by this
correction.
