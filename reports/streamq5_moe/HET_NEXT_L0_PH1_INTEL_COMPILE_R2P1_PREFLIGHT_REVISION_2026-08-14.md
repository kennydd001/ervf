# PH1 Intel compile-only R2P1 — TEMP-isolation repair

Status: frozen preflight-only revision; no compiler, device, payload or preflight call performed during construction.

The immutable R2P preflight result is 6/7 negative, SHA-256 `c474d2c28cae595c35647ea542df62ed7ca27009c0ca95d71d5251c7b8bb6860`. Its sole false conjunction is `actual_r2_transaction_simulation`; lexical source, emulator/mutations, AST contract, bindings and absence gates are true.

Read-only diagnosis: R2P redirected `base.OUT`, `base.FAILED` and `base.QUARANTINE`, but the inherited production recovery function discovers stale directories through `base.REPORTS.glob(...)`. Because `base.REPORTS` still pointed at the real report directory, the deliberately created TEMP stale directory was not found.

R2P1 changes only the test harness: within a `try/finally` it redirects `base.REPORTS` and every related runner/base output global to the TEMP root, runs valid-commit, already-complete, stale-temp, corrupt-final and immutable-failure cases, then restores every original global and verifier callable even on assertion/exception. It binds the immutable failed R2P result and prior R2P/R2/R1B chain. The R2 OpenCL source, backend and physical runner are unchanged.
