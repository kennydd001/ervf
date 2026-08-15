# PH1 Intel execution R7C — closed outer-failure and verifier-independence revision

Date: 2026-08-14

R7C preserves the immutable R7A physical computation and the R7B authorization-result gate. It changes no payload, kernel, buffer, launch, numerical gate, resource threshold, expected output, or claim.

R7C is closed (`execution_open=false`, token `PENDING`) pending an independent source audit and a separately authorized no-device preflight.

## Exact repairs

1. The R7A authorization-preflight result is hash-bound and parsed before any configuration, recovery, payload, or OpenCL path. The exact seven check names must all be true, with PASS 7/7, `no_payload_compiler_device=true`, exact R7A ACK, and exact R7P hash.
2. An R7C outer boundary surrounds the complete immutable `execute_authorized` call. Exceptions from psutil import, start-RAM sampling, payload construction, post-payload sampling, predevice setup, delegated device execution, and serialization/commit produce a bounded create-new structured R7C failure bundle. A valid physical commit is returned and never polluted. Stale R7C failure temporaries are quarantined and abort the attempt.
3. The R7C verifier does not import R7B/R7C candidate runners. It freezes its own paths, constants, hashes, lock schema, authorization-result contract, and extension contract. Only after that extension passes may it import the hash-bound frozen R7A numerical verifier.
4. The no-device preflight must exercise the actual R7C outer failure/transaction functions in a TEMP root for every seven declared stage labels, oversize bounding, stale quarantine, and valid-commit nonpollution. It must invoke the actual independent extension validator and prove all frozen negative mutations fail.

Claim remains limited to one real expert/input Intel correctness component. No performance or model-level claim is opened by this revision.
