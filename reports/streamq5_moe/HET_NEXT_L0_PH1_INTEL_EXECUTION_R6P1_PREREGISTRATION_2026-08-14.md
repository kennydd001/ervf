# PH1 Intel execution R6P1 — synthetic preflight consistency repair

Status: closed/PENDING. No preflight, payload, compiler, OpenCL load, or device
call is authorized by this file.

R6P1 binds the frozen R6 implementation, R6P source/lock/audit, immutable R6P
12/15 result, and its independent diagnosis. It changes only the three false
synthetic checks:

- no-device proof is AST import/call inspection, without literal self-scan;
- the quantizer fixture expects exact ties-to-even q values
  `[-15,-7,-4,0,4,7,11,15]`;
- verifier prepared weights are decoded independently from the q=1/scale=1
  packed record bytes, while input and outputs remain zero.

Both fixed-width sentinels remain `(1,512)` and `(1,2048)`. The full verifier
fixture remains production-shaped, uses exact BUFF/ARGS/LAUNCH and full counter
payloads, first requires every baseline `verify_dict` conjunct to pass, then
requires every frozen mutation to make the real verifier fail. Production
runner/backend/common/verifier/kernel/codec/science are unchanged.
