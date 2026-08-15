# HET-NEXT-L0 PH1 NVIDIA NC11 compile-only preregistration

Status: immutable design-only; all implementation and execution remain closed. NC11 supersedes NC10 only for the four blockers in independent audit SHA `00583f27893bcab82ffdf1e7e97cff32c2e2ed2300d8edb99a9b7accb300fcfd`.

The canonical fixture-manifest file is raw UTF-8 with exactly one leading `EF BB BF`. Its raw size, including those three bytes, is subject to the 8 MiB cap. The shared bounded loader stats first, rejects zero/over-cap without open, otherwise reads exactly the raw size, requires and strips exactly one BOM, strictly decodes UTF-8, then parses JSON once. Missing, double or wrong BOM and schema failure are explicit fixtures. Accepted cap-minus-one and cap fixtures are BOM + complete sentinel JSON + ASCII-space padding and have frozen full SHA-256 values.

Every pre-case loader rejection has ten ordered NVRTC ledger rows with `attempted=false`, `compiler_loaded=false`, `attempt_consumed=false`, no publication, and permits a later corrected invocation. It contains no compiler evidence.

Fresh topology requires absence of all inherited NC8/NC9/NC10 durability and preflight-failure/quarantine roots, every NC11 output/failure/quarantine root, and the current NC11 durability root. NC11 durability has an exact schema plus absent, file, directory, orphan, oversize and collision fixtures. NC11 independent-verification positive, negative, failure and quarantine paths are explicitly closed.

All NC10/NC9 requirements otherwise remain unchanged. No run is authorized.
