# PH1 Intel execution R7C2 — exact device-state and clean-state repair

Date: 2026-08-14

R7C2 is the immutable two-fix successor to R7C1. It changes no authorization-result gate, physical R7A computation, payload, OpenCL behavior, numerical gate, resources, output, or claim. It is closed pending audit.

1. Every parseable inherited R7A failure retains `inherited_device_opened` exactly when it is a JSON Boolean. A valid inherited failure requires that Boolean. The R7C2 top-level delegated summary derives `device_opened` only as the OR of retained exact inherited values, never from disposition. Missing/malformed/oversized evidence uses `null` and cannot become a valid inherited bundle.
2. Static clean-state adjudication covers the real R7A physical output/failure/quarantine paths, the separate R7C2 revision output/failure/quarantine paths, every matching in-progress glob, the required immutable R7A authorization result, the required R7P result, and the absent R7C2 preflight/verifier results. No hidden stale path is permitted.

The no-device preflight additionally exercises a valid inherited `device_opened=true` return-3 case plus wrong-type and false-value mutations. All R7C1 lifecycle cases remain required.

Claim remains one real expert/input Intel correctness component only.
