# NC19I1 source implementation design

The package consists of four substantive, fresh files: an import-inert stdlib shared contract; a direct NVRTC runner; a no-device/no-payload static preflight; and a standalone verifier that imports no candidate module. The runner and preflight call the same classifier, compiler state machine, cache/environment functions, and transaction/failure functions.

The classifier consumes raw Base64 observations as sole content authority, checks byte count/SHA/strict JSON projection/typed schema/identity policy, authenticates exact source-lock bindings, applies stage-specific required/absent sets, validates in-progress patterns, and permits at runtime the required inputs plus exactly one declared terminal. Unknown terminal identifiers, mixed terminals, collisions, orphans, traversal, cap violations, and stale source-lock maps are rejected.

The verifier caps before reads, confines candidates to canonical positive/negative or precommit staging roots, reconstructs the three-file commit envelope and raw artifacts, requires exact PTX directive and two-entry cardinality, parses bounded ELF64 symbol tables for exactly the two kernels, checks every ledger/ABI/ownership/cleanup/module/cache/environment/provenance cross-link, and independently authenticates locks plus preflight evidence. A postcommit verifier protocol-negative is written only to its separate bounded failure namespace.

Frozen boundary: source implementation and read-only syntax/hash/absence evidence only. No preflight, NVRTC/compiler, scientific payload, nvcuda/Driver, CUDA Runtime, or device call has been made.
