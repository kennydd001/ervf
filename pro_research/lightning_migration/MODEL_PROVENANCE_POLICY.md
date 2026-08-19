# Model provenance policy — Nemotron 3.5 Lightning migration

## Non-negotiable rule

A directory name is not a model identity.

The following are **not** sufficient:

- `models/nemotron_3_5_lightning`;
- `models/nemotron_3_5_lightning_v35`;
- a five-shard layout;
- 52 layers, hidden size 2688, 128 routed experts and top-6;
- `model_type = nemotron_h`;
- a result file that merely stores one of those paths.

Nano and Lightning share much of this macro chassis.

## Required confirmation

The migration runners proceed only when all of these are present:

1. acquisition provenance for the official repository  
   `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`;
2. an immutable resolved Hugging Face revision;
3. the expected 30B-A3B macro contract;
4. at least one structural MTP marker in config or tensor inventory;
5. at least one structural LatentMoE marker in config or tensor inventory;
6. no Nano identity marker or conflicting acquisition record.

The official acquisition manifest is written by
`acquire_lightning.py`. README/model-card words are supporting evidence only; they never satisfy a structural marker gate. `model_guard.py` is fail-closed.

## Classifications

- `lightning35_confirmed` — allowed to execute migration tests.
- `lightning_claim_unverified` — path/name claim only; no GPU test.
- `nano_v3_confirmed` — preserved as Nano evidence; no Lightning claim.
- `identity_conflict` — stop immediately.
- `unknown` — stop immediately.

## Legacy-result treatment

A legacy result is not deleted.

- Hardware-only measurements can be retained with their exact environment.
- Generic correctness/harness fixes remain engineering knowledge.
- A checkpoint-specific result is tagged Nano, Lightning-confirmed, ambiguous,
  DeepSeek, or technical-incomplete.
- `technical_failure` and `instrumentation_complete=false` mean
  **INCOMPLETE**, never `False`.
- Every quality, route, activation, cache, expert, state, MTP and end-to-end
  timing verdict requires confirmed checkpoint provenance.

## Why the old `_v35` evidence is revalidated

Several V3–V6 result JSONs record a `_v35` directory and are likely valuable.
They still do not contain an immutable official repository/revision fingerprint.
The migration therefore tries to link their metadata and source hashes to the
new confirmed checkpoint. A successful linkage preserves them; otherwise they
are rerun. They are not silently discarded and not silently trusted.
