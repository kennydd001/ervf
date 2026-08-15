# HET-NEXT-L0 PH1 Intel execution R8V1-R1 verifier erratum

Date: 2026-08-14. Status: frozen closed source; no execution until independent audit and a separate authorization-only revision. No payload construction, compiler, OpenCL or device operation is permitted.

R8V1-R1 supersedes only the closed R8V1 verifier source. It binds the R8V1 independent audit SHA-256 `0863b759eaa5a2fa6eaca0a7c24d3dcec5dfe5454eee9c66ab2699a36719f587`. The R8A5 physical bundle and first verifier remain immutable.

## Exact prior-verifier classification

The prior R8A5 verifier JSON is accepted only at SHA-256 `d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917` and only with its exact nine top-level fields. It must have kind `ph1_intel_execution_r8a5_independent_verification`, `pass=false`, `passed=27`, `total=29`, `terminal_state=invalid`, `terminal_valid=false`, and the exact frozen claim. Its check map must have exactly 29 frozen names, exactly `topology` and `terminal_contract` false, and all other 27 true. Its mutation map must have exactly the frozen 31 names and all values true. An added, missing or changed field—including a fabricated model flag—is rejected. A nonvacuous mutation suite changes each structural class and requires rejection.

This prior outcome is classified only as `prior_verifier_outcome="verifier_protocol_negative"`. It is not a scientific or physical negative. R8V1-R1 separately reconstructs the immutable bundle and records `bundle_adjudication="positive"` only if every current topology, provenance, bundle, authorization, numerical and direct physical check is true.

## Exact closed topology

The closed-source expected family is a literal, case-preserving set: the eight frozen R8A5 entries, the old R8V1 preregistration/lock/audit, and the new R8V1-R1 preregistration/lock. The not-yet-existing R8V1-R1 audit and output are absent. Directory enumeration is authoritative; Windows prefix-glob observations are retained only as diagnostics and are compared against their own exact expected sets. Casefold collisions, missing entries, uppercase/lowercase extras, orphan, temp, failure and quarantine names are all rejected in a non-writing mutation suite. A later auth-only revision must add exactly its source-audit path and retain the fresh output absence gate.

## Positive adjudication

The verifier retains the R8V1 independent three-file bundle, R8A5 authorization, exact 20-check R7A numerical reconstruction, 18 physical gates, Intel identity, operation/ownership/buffer/argument/launch/read/release/cleanup cardinalities, 22 controls, resources, forbidden-zero counters, five output hashes, and the 31-case terminal mutation replay. The immutable bundle hashes are result `9d1ac21f...`, manifest `2d13137f...`, commit `07d9f03e...`.

The canonical R8V1-R1 output may be written create-new only when every current check is true. It then has `terminal_state="positive"`, `terminal_valid=true`, `pass=true`, `prior_verifier_outcome="verifier_protocol_negative"`, and `bundle_adjudication="positive"`. On any current failure it writes no canonical output and returns nonzero. It never emits a canonical negative or partially passing bundle. The claim remains one real expert/input Intel correctness component only.
