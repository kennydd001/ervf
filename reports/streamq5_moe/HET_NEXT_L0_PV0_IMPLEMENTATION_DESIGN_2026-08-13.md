# HET-NEXT-L0-PV0 — implementation and independent-verification design

## Present state

This document accompanies the PV0 preregistration and is non-executable. No PV0
runner, lock, preflight, output directory or device action exists. Earlier C0
and CAP0P source lineages are not inherited. Only the small, independently
audited CAP0X-R2 process-isolation and established Intel host-USM/NVIDIA staged
Q5 mechanisms are allowed as source material, and their exact copied portions
must be re-audited in PV0.

## Minimal future file split

A future implementation should stay auditable by using six standalone files:

1. `build_het_next_l0_pv0_source_oracle.py`: CPU-only sealed p0 reader,
   33-tensor source reader, quantizer and independent source/Q5 oracle builder.
2. `run_het_next_l0_pv0_intel_child.py`: ranks 0-3 only; no safetensors import.
3. `run_het_next_l0_pv0_nvidia_child.py`: ranks 4-9 plus shared only; no
   safetensors import.
4. `run_het_next_l0_pv0_coordinator.py`: process launch, barrier, raw capture,
   official-order merge and atomic result/failure transaction.
5. `verify_het_next_l0_pv0_independent.py`: no import from files 1-4; independently
   rereads/requantizes/replays all 33 tensors and all metrics/controls.
6. `preflight_het_next_l0_pv0_static.py`: source/lock/AST/schema and TEMP-only
   process-protocol simulation; never imports a device library or opens D2/shard.

Runner and verifier locks start `execution_open:false` with a pending audit token.
The preflight source itself is hash-bound. Output paths are absent at freeze.

## Phase state machine

### Phase 0 — static only

Check all immutable byte bindings, exact route constants, ownership map,
ascending-ID merge permutation, Q5 byte arithmetic, output absence and forbidden
imports/calls. Simulate create-new success/failure transactions, wrong nonce,
wrong owner, premature child exit, timeout and sealed p1 request in a TEMP
directory. This phase opens neither payload and loads no compiler/device DLL.

### Phase 1 — CPU source/oracle

After separate authorization, hash the complete official shard once, then use
read-only offsets for only the 33 permitted tensors. Materialize one projection
at a time, emit its canonical source/Q5 manifest and release it. Read only the
six p0 D2 tensors listed in the preregistration. Build:

- source-BF16 full-shape graph and strict D2 component equality;
- independent CPU-Q5/ERVG intermediates;
- two child input packages containing only assigned compressed records, p0
  activation rows/gather maps and frozen metadata;
- a sealed CPU-oracle bundle that neither child can open.

The child packages together contain exactly 22,167,552 Q5 payload bytes. They
may be anonymous named mappings or create-new temporary files, but must be
read-only in children, byte-hashed before launch and destroyed after capture.
No second decoded-weight cache is allowed.

Phase 1 commits before device startup. Failure leaves immutable CPU-only failure
evidence and opens no child process.

### Phase 2 — process-isolated component run

The coordinator creates both children with unique nonce-bound argv and captures
stdout/stderr plus PID/creation-time identities. Children parse a fixed package
schema, validate ownership and digests, initialize only their own backend,
declare ready, and wait on one coordinator barrier. Each runs every assigned
expert exactly once over its p0 full-shape gather plus its synthetic control;
NVIDIA also runs shared. There are no warmups, repeats or clocks used for a
performance statement.

Children serialize raw BF16 stage arrays and cleanup evidence before exiting.
The coordinator validates both result schemas, independently reconstructs the
expert-ID merge permutation, merges token 15, writes raw/result only after all
gates are computable, then cleans packages and writes the commit last. A child
failure triggers bounded termination/wait and atomic failure evidence; it never
promotes a partial positive result. A valid existing commit returns
`already_complete` without mutation.

### Phase 3 — independent adjudication

The verifier starts from the official shard, D2 raw and immutable locks, not
from builder summaries. It independently:

- verifies the p0-only read ledger and p1-p3 range exclusion;
- reconstructs all 33 source keys/shapes/hashes;
- requantizes and decodes all records byte-for-byte;
- reruns source-BF16 and Q5/ERVG full-shape graphs;
- checks exact expected raw tensor keyset, dtype, shape, bytes, finiteness and
  digest;
- recomputes per-stage and aggregate FP64 metrics and BF16 ULP diagnostics;
- rebuilds ascending-expert-ID merge states;
- reconstructs all safe/unsafe controls and queue-counter ordering;
- checks child identity, backend ownership, package access and cleanup.

The verifier exits nonzero on any false conjunct and writes a separate
create-new verification artifact. The coordinator cannot label the experiment
positive until that artifact is hash-bound and all independent gates pass.

## Frozen raw schema

For each selected expert retain, for source CPU, Q5 CPU and assigned device:

- exact hit token indices and top-k positions;
- `gate_bf16`, `up_bf16`, `silu_bf16`, `activation_bf16`, `down_bf16` for all
  p0 hits;
- token-15 route weight word and weighted-down vector;
- device record digest, pointer class, call sequence and ownership tuple.

Also retain ten routed accumulator states in ascending-ID order, shared gate,
up, SiLU, activation, raw, outer-sigmoid word, gated output, and the final
token-15 routed/shared component vectors. Each tensor has canonical name,
little-endian dtype, exact shape, byte count and SHA-256. Compiler source/log,
backend identities, package manifests, read ledger, process ledger, control
ledger, resource samples and cleanup ledger are JSON evidence. Compressed Q5
payloads and source tensors are not retained in the final bundle; the verifier
reconstructs them from the pinned shard.

## Exact ownership and merge table

The child assignment is rank-based:

- Intel: `(rank,expert) = (0,50),(1,199),(2,237),(3,474)`;
- NVIDIA: `(4,245),(5,374),(6,239),(7,8),(8,168),(9,12)` and shared.

The only legal merge order is:

`(expert,rank) = (8,7),(12,9),(50,0),(168,8),(199,1),(237,2),`
`(239,6),(245,4),(374,5),(474,3)`.

The verifier derives this permutation from the raw route IDs and rejects a
stored permutation that merely repeats these constants without matching the
captured route.

## Backend contracts

Intel's input and assigned Q5 codes/scales must be directly CPU-populated
host-USM. Kernels receive those pointers only via
`clSetKernelArgMemPointerINTEL`; audited call counters for `clCreateBuffer`,
`clEnqueueWriteBuffer`, copy and migrate operations must remain zero for weight
payload. NVIDIA may use the established D7/Next staging path, but all staging
bytes and source pointer ranges are listed; its device-produced activation is
the direct input of its down kernel. Neither child may consume CPU oracle stage
arrays.

Both backends freeze width 8, group 128, the same Q5 decode and the same ERGV
reduction tree. Linear BF16 endpoints are required bitwise against the CPU-Q5
oracle. The only tolerance enters after device SiLU, as specified in the
preregistration. A hidden CPU activation roundtrip, alternate GEMV library,
dense dequantized-weight materialization or unlisted payload is
`invalid_protocol`.

## Deliberate limitations and next decision

PV0 uses a previously inspected p0 row, not a fresh input. It cannot validate
route generalization, p1-p3, full-layer behavior or speed. Its value is to
separate three questions that were previously entangled: real official-weight
Q5 reconstruction, device-specific whole-expert correctness, and deterministic
cross-process ownership/merge.

If PV0 is independently positive, the next experiment may preregister fresh
held-out p1-p3 correctness. Performance remains a later, separate protocol and
must not reuse PV0's single execution as timing evidence. If PV0 is negative,
the retained per-stage arrays localize the first divergence to codec, Intel
linear, NVIDIA linear, activation, down, merge, shared or control without a
threshold change or device retry.
