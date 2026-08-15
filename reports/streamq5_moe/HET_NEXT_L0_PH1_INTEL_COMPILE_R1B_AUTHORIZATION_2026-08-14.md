# PH1 Intel compile-only R1B — authorization-only repair

R1B is an immutable new authorization revision. It changes no OpenCL source byte, build option, device rule, binary gate, lifecycle rule, threshold or claim. It repairs only the R1A authorization-preflight defect: the preflight now proves its behavior through AST import/call inspection instead of searching its own blacklist strings, and it explicitly verifies its own SHA against the R1B lock.

R1B binds the complete immutable R1A source/authorization set and closed R1 static PASS 8/8. No separate local R1A audit-report artifact existed at freeze; therefore the actual R1A backend, runner, auth-preflight, auth document and open-lock hashes are all bound directly. The known R1A verdict is limited to the deterministic preflight self-match and missing self-hash check; its runtime authorization was independently found correct.

The unchanged source is 7,909 bytes, SHA-256 `06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528`. The frozen physical acknowledgement is `PH1_INTEL_COMPILE_R1B_AFTER_PREFLIGHT_PASS_AND_INDEPENDENT_FINAL_AUDIT_GO`. The new output is `reports/streamq5_moe/het_next_l0_ph1_intel_compile_r1b` and must be absent before execution.

This source freeze performs no preflight, compiler, payload or device call. The claim remains compile-only eligibility and nothing more.
