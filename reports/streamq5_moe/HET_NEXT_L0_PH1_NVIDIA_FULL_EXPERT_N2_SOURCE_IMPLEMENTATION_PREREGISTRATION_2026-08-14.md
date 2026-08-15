# HET-NEXT-L0 PH1 NVIDIA full-expert N2 source implementation preregistration

Status: **execution closed**. This immutable N2 supersedes the N1 implementation after the independent N1 source-audit (`2a7798ef…`). No payload, NVRTC compilation, CUDA Driver call, static preflight, or physical execution is authorized by this source freeze.

N2 preserves the approved N1 arithmetic and one-real-expert/input claim. It changes only evidence, verification, terminal classification, and failure handling:

1. The static preflight AST-audits all 30 Driver ABI bindings (argument and return types), 64-bit device pointers, launch pointer slots, exact 9/5/4/9/1/7 schedule, context ordering, no post-release Driver query, and inert loader call surface. It executes current transaction, release-fault, kernel mutation, and production verifier-contract mutations without payload/compiler/device access.
2. Compile verification binds the candidate `source.cu` byte hash to the frozen source lock and authorization, independently checks the width-8 DAG, and rejects PTX/SASS FTZ, approximate, unresolved, or unexpected entrypoint evidence. Compiler failure evidence always contains the complete ten-operation intended ledger with an explicit `not_attempted` suffix and destroy disposition.
3. Physical verification independently checks the exact NVCUDA loader identity, all ABI return types and return codes, context/stream/allocation/pointer/argument crosslinks, exact seven `cuMemGetInfo_v2` ledger rows, operations, resources, 30 releases, and runtime loaded-module evidence. Forbidden-call evidence is a hash-bound static callgraph plus observed loaded modules; it is not an instrumentation claim.
4. A committed device-numerical negative is valid only when one or both of `stages_exact` and `counters_exact` fail while every protocol, control, identity, ABI, schedule, resource, forbidden-surface, lifecycle, and cleanup gate remains true. Any other false gate is an invalid infrastructure/protocol failure and cannot be committed as scientific evidence.
5. All 22 predevice controls retain requested and presented metadata plus the exact checker-stage trace. The independent verifier reconstructs every mutation, including the wrong-LUT case, and compares the full evidence rows.

The source lock remains closed with pending tokens. Only `py_compile` of Python source is permitted before independent source audit.
