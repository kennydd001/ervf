# HET-NEXT-L0 PH1 NVIDIA N4 source freeze

Status: **closed; no execution authorization**. N4 supersedes N3 and binds independent N3 audit SHA `9b326722f4d61a083fb7bbd991f3ea6d00b717962d2ea6748afdbc6ecf2560bb`.

N4 preserves the frozen arithmetic/kernel and narrow one-expert/input correctness claim. The static preflight treats the 171,696,126-byte D2 raw and 3,999,619,288-byte official shard as frozen absolute-path/SHA/size identities and permits only `stat`; its explicit payload-byte counter must remain zero. All smaller bindings are rehashed directly. Production-verifier fixture provenance is injectable and cannot hash payloads.

The isolated fixture child is a separately frozen script invoked as exact `python -I -B <absolute-child>`. It bootstraps the preflight, transaction, common, backend, and independent verifier solely by frozen absolute paths; stdout, schema and exit code are exact. Fixtures exercise current transaction/bundle helpers and actual `verify_compile`/`verify_physical` entrypoints with injected payload-free provenance, plus mutated terminal, ABI, pointer, resource and bundle evidence.

Compile verification binds candidate CUDA bytes to the source lock and checks the independent width-8 DAG plus PTX/SASS no-FTZ/no-approx/no-unresolved contract. Physical verification additionally requires exact pinned-write hashes/pointers, all Driver ABI vectors and return codes, owner TID, context/module/function/stream/allocation/argument links, meminfo sequence, releases, resources and pre/post runtime-module evidence. Only stage/counter mismatches can be scientific negatives. Compiler, context, ownership, cleanup, atomic create/publish/failure and corruption cases are fail-closed and bounded.

No preflight, payload, compiler or device call has been performed. Before independent source audit only `py_compile` and read-only small-file hashing/output-absence checks are allowed.
