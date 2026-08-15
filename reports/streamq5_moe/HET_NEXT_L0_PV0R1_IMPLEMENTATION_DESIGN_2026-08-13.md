# HET-NEXT-L0-PV0-R1 — bounded implementation design

## Non-executable state

Only design exists. No PV0-R1 runner, verifier, preflight, lock or output path is
authorized. The original PV0 documents remain immutable and are not imported.
Future implementation begins only after an independent design GO.

## Six standalone future components

1. A CPU builder with a header-only D2 range reader, 33-tensor shard reader,
   graph-matched source control and independent CPU Q5/ERVG oracle.
2. An Intel child containing only the audited host-USM backend and PV0 kernels.
3. A NVIDIA child containing only the audited D7/Next staged-Q5 backend and PV0
   kernels.
4. A standalone Win32 coordinator implementing framed pipes, job ownership,
   bounded lifecycle and create-new transactions.
5. An independent verifier that imports none of components 1-4 and independently
   rereads/requantizes/replays source, codec, metrics, merge and controls.
6. A static preflight that opens no D2/shard and imports no device library; it
   checks AST/contracts and exercises the actual framing/timeout/transaction
   functions in TEMP with mock children.

Locks begin closed and bind both R1 documents, all provenance, sources, compiler
inputs, verifier and preflight. A later authorization revision may only change
open/token/output paths after source audit.

## Phases

### Static phase

Assert output absence, source hashes, exact route/hit/ownership/merge tables,
22,167,552-byte Q5 arithmetic, host-USM byte totals, raw-schema byte bound and
forbidden imports. Exercise partial frame, oversize, wrong nonce/sequence, child
crash, ready/result/exit timeout, job termination, stale output, valid commit,
failed verifier and p1 range rejection. No payload or backend opens.

### CPU phase

Read the D2 header plus only six allowed ranges. Separately verify the full
official shard SHA, parse its header and stream only the 33 allowed projections.
Construct graph-matched source and independent Q5/ERVG evidence. Determine both
control witnesses and commit their coordinates/words before child launch.
Produce read-only Intel/NVIDIA packages totaling exactly 22,167,552 Q5 bytes;
packages include only inputs, gather maps and assigned record manifests.

The CPU phase fails before devices unless the D2 route/hits, source graph
envelope, codec and resource gates pass. It does not create a positive result.

### Device phase

The coordinator launches fresh suspended children, assigns them to one job,
resumes, verifies both ready frames, then sends start. Intel computes 12 selected
hits for four experts; NVIDIA computes 11 selected hits for six experts plus 16
shared rows. Each child retains raw stage arrays, pointer/call ledger, compiler
source/log/binary hash, device identity, resource samples and cleanup. No timing
sample or repeat exists.

After zero exit and no survivors, coordinator performs only the frozen
expert-ID-sorted token-15 merge. Candidate raw/result stay temporary.

### Independent phase and commit

The verifier receives no trusted summaries. It rebuilds the D2 allowlist from
the 170,656-byte header, checks every completed range, rereads and independently
requantizes all 33 shard tensors, rebuilds exact hit shapes, source graph,
CPU-Q5 oracle, device metrics, merge sequence and both control witnesses. It
requires the exact raw tensor schema and recomputes every digest/metric.

Only a verifier exit-zero artifact whose source hash is lock-bound permits
atomic promotion. Raw and result promote first, verifier artifact next, commit
last. Any failure cleans/quarantines packages and temporaries and records every
disposition; it cannot retry.

## Raw schema and first-divergence localization

For each of ten selected experts and each implementation `source`, `cpu_q5` or
the owning device, retain exact hit pairs plus BF16 gate, up, SiLU, activation,
down. Token-15 adds route-weight word, weighted down and, in ascending expert-ID
order, accumulator-after-add. Shared retains full16 gate/up/SiLU/activation/raw,
the 16 outer gate words and gated outputs. Also retain official D2 token-15
routed/shared references and all input/route tensors used.

Each tensor manifest entry is independently recomputed as canonical key, dtype,
shape, little-endian bytes and SHA-256. The report names the first failed stage:
source graph, codec, Intel gate/up, Intel activation/down, NVIDIA gate/up,
NVIDIA activation/down, shared, official-order merge, quality or control.

## Key scientific boundaries

- Tokens 0-14 are used only to preserve selected-expert gather shapes. Their
  selected-subgraph outputs are retained but never compared with the full D2
  routed aggregate.
- Token 15 has exactly the ten selected experts, so its selected aggregate is
  complete. Comparison to official D2 is tolerant and explicitly diagnostic,
  never bitexact.
- Gate/up exactness tests the frozen Q5 decode/reduction. Once SiLU may differ,
  SiLU/activation/down/merge use the preregistered 0.001 envelope. Q5-to-source
  quality remains the separate inherited 0.08 envelope.
- Intel output is directly read host-USM, so the no-copy rule is executable;
  NVIDIA staging is fully enumerated. Neither path may substitute CPU stage
  arrays.
- Process isolation is a new audited harness. CAP0X-R2 proves only that the two
  backend classes previously coexisted without process error.

## Decision after PV0-R1

A positive result justifies a new held-out p1-p3 correctness preregistration; it
does not open performance. A negative retains enough raw evidence to repair one
localized mechanism without reusing the physical attempt or changing a frozen
threshold.
