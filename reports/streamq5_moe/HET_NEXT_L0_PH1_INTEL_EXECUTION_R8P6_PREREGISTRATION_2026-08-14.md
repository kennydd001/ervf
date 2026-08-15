# PH1 Intel execution R8P6 — consolidated current-writer CPU preflight

Date: 2026-08-14  
Status: immutable closed preregistration; no execution authorization.

R8P6 supersedes R8P5 and binds its independent audit SHA-256 `c431578cc6a1edefa0d3843ca0fdd26ec5d07b9e592d5a758a4ed0f40e36d608`. This is a consolidated no-device revision. The exact current R8P6 `atomic_create`, `verify_bundle`, `publish`, cleanup, and quarantine helpers are shared by the success path and the seven-outcome TEMP transaction suite. The independent verifier has its own current output writer and six-outcome TEMP suite. Both require exact nonempty key sets and all values true; empty, missing, extra, and false transaction dictionaries are rejected.

The R8P5 three-state CPU-slice provenance and five failure-lifecycle gates are retained. The exact state mapping remains `not_started → false/false`, `started_not_completed → true/false`, and `completed → true/true`. State validation now proves `isinstance(state, str)` before set membership. Unknown string, list, dictionary, integer, null, contradictory Booleans, missing state, and extra state are all rejected without raising from the validator.

All dual venv/base identity, runtime, RECORD, 16-GiB RAM, frozen CPU preparation, 22-control, five-stage-hash, R7D1/R8P1 provenance, failure, static callgraph, topology and claim-boundary gates remain unchanged. Invalid argv/token returns before filesystem mutation. A later source GO may authorize only CPU reading of the frozen preparation slice; model forward, compiler, OpenCL, CUDA and device actions remain forbidden. R8P6 is closed/PENDING and may not be executed before independent source audit.
