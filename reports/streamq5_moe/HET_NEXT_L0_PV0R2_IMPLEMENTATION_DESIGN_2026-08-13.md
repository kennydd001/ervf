# HET-NEXT-L0-PV0-R2 — implementation design

## Non-executable state

No PV0-R2 runner, verifier, preflight, lock or output path exists. This design
binds the R2 preregistration and 33-record machine-readable manifest. It carries
forward R1's selected 23-hit scope, header-only D2 seal, standalone process
harness, exact controls and 8 MiB evidence cap only as explicitly refined by R2.

## Mandatory standalone modules

Future implementation has separate CPU builder, Intel child, NVIDIA child,
Win32 coordinator, independent verifier and static preflight. Verifier imports
no builder/device/coordinator code. Children import no safetensors/Torch model
reader and can open only their nonce-bound read-only package. Preflight loads no
device library and opens no D2/shard payload.

## Builder phase

The D2 reader parses only 170,664 header bytes and six allowlisted p0 ranges. It
reproduces route IDs, weights and exact 23 hits. The official shard reader binds
the complete shard SHA once, parses the 194,000-byte header, requires exact
manifest equality, then streams the exact 33 ranges one at a time.

For each record it independently quantizes using FP32/ties-to-even rules, checks
all manifest digests and writes assigned codes/scales only. Source PyTorch graph
must reproduce full16 D2 routed/shared raw bitwise, proving the graph shape/order
before Q5. Independent Q5/ERVG arrays, selected control witnesses and expected
device raw schema commit before child launch. Builder exits; coordinator proves
its PID/creation identity gone before entering device phase.

## Device phase

Coordinator creates two suspended children, assigns both to a fresh
KILL_ON_JOB_CLOSE job, resumes and exchanges bounded nonce/role/sequence frames.
Intel uses only host-USM including outputs. NVIDIA uses exactly the R2 allocation
and copy table; the final source implementation must replace every prose size by
one machine-readable table and static-preflight sum. Device-generated activation
is the down input. Captured outer gate linear BF16 is data, not a weight; NVIDIA
computes/stores sigmoid BF16 and gate-first shared multiply.

Both children retain raw per-stage arrays, exact pointer/copy/call ledger,
compiler artifacts, device identity, resources and cleanup. Coordinator performs
only the ascending-ID BF16 token15 merge after both clean exits. No repeat,
warmup or performance timestamp is adjudicated.

## Verifier phase

After both children exit and device cleanup balances, launch a fresh independent
verifier. It reconstructs the D2 range manifest, 33 source records and exact
codec; reruns source PyTorch graph and Q5/ERVG; checks p0 D2 equality; rebuilds
all device metrics, shared outer-gate semantics, official-order merge, controls,
raw schema, process lifecycle and phase/resource intervals. It must derive the
NVIDIA allocation sum and reject any unlisted allocation/copy or inconsistent
table value.

Only its hash-bound positive artifact permits raw/result/verifier promotion and
commit-last. A negative verifier is immutable failure evidence, never a retry.

## Raw evidence and gates

Per expert/source/cpu-Q5/device retain exact hit coordinates and BF16 gate, up,
SiLU, activation and down; token15 weighted vectors and ten accumulator states;
shared full16 gate/up/SiLU/activation/raw, outer-gate linear/sigmoid and gated;
official D2 references. Recompute every tensor dtype/shape/bytes/SHA and FP64
metric independently.

Exact gates are: source graph full16 bitwise to D2; codec bytes exact; device
gate/up bitwise CPU-Q5; downstream device stages <=0.001 CPU-Q5; Q5 and device
component aggregates <=0.08 source; all finite; every process, pointer, copy,
resource, seal, control and cleanup gate true.

## Claim boundary and next gate

A positive PV0-R2 proves only the selected real-weight process-isolated component
on known p0/n16. Held-out p1-p3 correctness requires a new preregistration.
Performance remains closed.
