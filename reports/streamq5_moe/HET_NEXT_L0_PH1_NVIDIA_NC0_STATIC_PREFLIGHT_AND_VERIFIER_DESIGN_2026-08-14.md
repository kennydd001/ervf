# NC0 static preflight and independent verifier design

## Static preflight

The standalone preflight is CPU-only and never imports the future compiler runner/backend. It AST-parses their frozen source and requires:

- direct `ctypes.CDLL` loading of only the pinned NVRTC DLL with cdecl convention;
- exact argtype/restype vectors for all ten NVRTC APIs and pointer-width checks;
- exactly one create/compile/log/PTX/CUBIN/destroy call surface and the frozen option tuple;
- no Driver/runtime/device/payload imports, paths, symbols or calls;
- authorization before source read, recovery, NVRTC load or output mutation;
- complete attempted/not-attempted and destroy ordering in success, nonzero-code, null-return and host-exception control flow.

Executable device-free fixtures use a fake NVRTC library against the actual future compiler function. They cover success and, at every API boundary, nonzero return, ctypes exception, null create, empty PTX, empty CUBIN, non-ELF CUBIN and destroy failure. Each fixture asserts exact ten-row suffix semantics and cleanup. Actual transaction functions are exercised in a temporary directory for clean success, verifier rejection, pre/post-link failure, fsync failure, stale/corrupt quarantine, valid-repeat, bounded writer failure and non-overwrite.

The preflight self-binds its source and lock, requires all NC0 targets absent, writes one create-new result only after every check, and reports `no_payload=true`, `no_driver=true`, `no_device=true`, `no_nvrtc_call=true`. A closed design lock prevents running it until a later implementation audit opens a fresh authorization revision.

## Independent compile verifier

The verifier imports no candidate implementation. It independently:

1. rehashes every direct small-file binding;
2. verifies the exact seven-file bundle, canonical manifest and commit-last hashes;
3. requires `source.cu` byte equality and SHA equality with the frozen N5 source and authorization observation;
4. checks result kind/status/claim, NVRTC DLL/header identities, cdecl ABI vectors, version 13.3, exact ordered options, and exact ten-row ledger/destroy semantics;
5. verifies build-log/PTX/CUBIN lengths and hashes and ELF magic;
6. parses PTX for exactly the four frozen entrypoints and rejects `.ftz`, approximate instructions/options, unresolved extern functions and unexpected entrypoints;
7. rejects any Driver/runtime/device/payload evidence or artifact.

Its preflight mutation suite constructs a complete valid-shaped synthetic compile bundle, proves the production verifier passes it, then independently mutates every result field, ledger row, source byte, artifact byte/count/hash, manifest entry, commit hash, option, ABI field, PTX entrypoint/FTZ marker, CUBIN magic and topology; every mutation must fail.

Failure verification requires exact provenance, bounded size, the complete ten-row ledger, correct attempted/not-attempted suffix and destroy disposition. It never upgrades a failure to a compile pass.
