# HET-NEXT-L0 PH1 NVIDIA full-expert N3 source freeze

Status: **execution closed**. N3 supersedes N2 and binds the independent N2 source-audit SHA `c1bbcd0078b3b432e405760e938d4248c2e3a00e2bdb28c1af529a5f5b7feaa8`. No preflight, payload read, NVRTC compile, or NVIDIA operation is authorized.

The approved N1 arithmetic/kernel and narrow one-expert/input correctness claim are unchanged. N3 repairs only evidence and lifecycle contracts:

- all three locks directly bind every source, lock, design/prereg/audit, CPU freeze item, Intel R8A5 item, native loader/header, D2 raw identity and official shard identity they rely on;
- the preflight validates exact complete Driver/NVRTC ABI vectors and mutations, exact schedule/pointer/context source structure, current transaction/failure cases, and device-free isolated production verifier/bundle path fixtures;
- compile verification binds the candidate CUDA source to the frozen lock and independently checks the width-8 DAG plus PTX/SASS no-FTZ/no-approx/no-unresolved evidence;
- physical verification requires exact ABI vectors, all return codes, owner-thread/context/module/function/stream/allocation/argument crosslinks, seven meminfo rows, full runtime module paths before load, after load, and after execution, and independently reconstructed 22-control requested/presented/checker traces;
- correct or incorrect closed/drifted authorization is mutation-free; `psutil` is deferred until after authorization;
- post-backend protocol/precommit exceptions retain complete backend evidence and `device_opened=true`; only stage/counter mismatches can be valid device-numerical negatives;
- compiler ctypes exceptions retain `attempted=true` at the failing operation and a complete explicit `not_attempted` suffix; cleanup and failure evidence stays bounded.

Only Python `py_compile` and read-only hashing/absence checks are permitted before independent N3 source audit.
