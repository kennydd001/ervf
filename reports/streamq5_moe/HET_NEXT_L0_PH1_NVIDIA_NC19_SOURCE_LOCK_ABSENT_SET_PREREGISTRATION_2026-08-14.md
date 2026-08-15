# HET-NEXT-L0 PH1 NVIDIA NC19 source-lock absent-set preregistration

Status: immutable design-only; implementation and every execution phase remain closed. NC19 binds NC18 audit SHA-256 `2a6d7ea64510e682da837a28c0fb05a723fa3fb90670f093154a41b988615f3f` and changes only its sole remaining blocker.

The sole observed NC19 source-lock document contains exactly 32 unique source identity entries. Its `expected_absent` is the exact sorted, unique 100-path projection of `descriptor.expected_absent_by_stage.implementation_freeze`. It excludes its own source-lock path and all exact 57 implementation-freeze required source, lock, provenance, and authorization-bootstrap inputs. Equality is exact and the required/absent intersection is zero.

The manifest retains the canonical source-lock bytes, Base64, byte count, SHA-256, and the bootstrap document that binds that identity. Negative cases mutate the actual observed source-lock bytes and bootstrap-bound identity: missing root, extra required path, self path, duplicate, unsorted order, wrong path, explicit 99 and 101 counts, and required-path intersection. Each is rejected. No preflight, compiler, payload, Driver, CUDA, or device run is authorized.
