# NC19I2 source implementation design

Four fresh substantive files form NC19I2: a stdlib-only import-inert shared contract, a direct NVRTC runner, a static fake-API preflight, and a standalone verifier. NC19I0/NC19I1 files and the corrected 1,106-case manifest are immutable inputs.

Tree digests use the manifest-declared eight-field order. The production evaluator dispatches root/unit cases separately from composition/runtime topology, applies byte-backed schema/identity rules, and executes all 1,106 expected cases. Runner and preflight use the same classifier, compiler state machine, environment/cache, terminal, transaction, recovery, and failure functions.

The independent verifier imports no candidate module. It caps each file before reading, rebuilds compile checks and artifact sizes, parses PTX and ELF symbols, and validates exact ABI, invocation, authorization, provenance, module timing, ownership, cleanup, environment, cache, and commit cross-links. Its canonical output path equals the locks. A distinct compile-valid-negative fixture proves negative adjudication; corruptions of both terminal classes must fail.

Freeze boundary: source, docs, closed locks, read-only hashing/AST, `py_compile`, and output absence only. Execution remains closed.
