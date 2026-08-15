# PORT80B-T0 official layer-0 gate design audit

Date: 2026-08-13  
Mode: CPU/source audit only; no reference forward or GPU action

## Verdict

T0-R2 followed by T0-P2 is the smallest honest gate that moves PORT80B from a
strong synthetic transport result to target-model evidence. It uses official,
differentiated layer-0 weights and natural top-10 routes. It can validate
router logits and layer-output fidelity, but cannot validate final vocabulary
logits or language quality. Those require the complete 48-layer checkpoint,
final norm and LM head in a later T1 gate.

## Reuse versus rebuild

| Asset | Decision | Reason |
|---|---|---|
| D9/D10 499+13 registration, cold-copy, stream ordering, cleanup and telemetry | Reuse structurally in T0-P2 | Already independently checked; bind to the new bank and routes. |
| STREAMQ5 group-128 Q5 quantizer and biased `q+15` 8-in-5 codec | Reuse after source lock and sentinel | The P1D builder/verifier establish exact wire semantics. |
| Width-8 Q5 kernels and intermediate-output capture | Adapt and reverify | Shapes match 2048×512/512×2048, but official source identity and natural top-10 must replace synthetic inputs. |
| N4A shape/byte math | Reuse as an arithmetic cross-check | It establishes 675,840 bytes/matrix and 2,027,520 bytes/expert, not real weights. |
| P0/P0C quality-split pattern | Reuse methodology only | The tested model was Qwen3-30B-A3B, not Qwen3-Coder-Next. |
| P4D routes and input IDs | Do not reuse as target evidence | They are top-8/128 routes from Qwen3-30B-A3B. |
| 49.9-GB uniform PORT80B bank and D10 synthetic outputs | Do not reuse numerically | Identical synthetic payloads cannot prove expert/layer identity. |
| Official shard-1 index, config and tokenizer | Reuse exactly at pinned revision | Shard 1 contains embedding plus all layer-0 tensors. |
| Layer-0 BF16 reference, natural routes and raw artifacts | Build fresh | This is the missing target truth source. |
| Real differentiated layer-0 Q5 bank | Build fresh | Exactly 513 official records; about 1.04 GB. |
| T0-P2 candidate runner and independent raw-byte verifier | Build fresh | Must bind candidate arrays to T0-R2 oracle and source identities. |

## Minimal claim-carrying scope

- One exact official shard: 3,999,619,288 bytes, SHA-256
  `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- Complete embedding plus decoder layer 0—not a router-only shortcut.
- Four frozen domains × 16 tokens; positions 8–15 held out: 32 primary rows.
- All 512 router logits, natural normalized top-10 IDs/weights, ten routed
  experts plus shared expert, and complete composed layer output.
- Official BF16 reference and independently quantized/dequantized Q5 reference.
- Raw tensors retained so a verifier can reproduce every hash and metric.

This scope can carry: “official real-weight layer-0 natural-routing and Q5
fidelity” and, after T0-P2, “exact official layer-0 Q5 expert-plane physical
execution.” It cannot carry a whole-model quality or throughput statement.

## Resource envelope

- Official shard already targeted: 3.9996 GB on disk, read-only/mapped.
- New real Q5 bank: 1,040,117,760 bytes; prefix 1,011,732,480; cold tail
  26,357,760; shared 2,027,520.
- Ten selected routed records: 20,275,200 bytes; shared: 2,027,520 bytes.
- T0-R2 gate: start RAM at least 8 GiB, reserve at least 2 GiB, peak process RSS
  at most 12 GiB. Shard tensor access must be selective; never materialize the
  whole BF16 expert plane plus all dequantized experts simultaneously.
- T0-P2: at least 512 MiB post-allocation VRAM and 2 GiB host-RAM reserve.
- Require 20 GiB free disk; the observed system had ample headroom at preflight.

## Current hard blockers and gate order

The immutable-input preflight now passes 14/14. Execution is nevertheless
closed until a separately hashed reference runner and independent verifier are
audited. The runner must demonstrate a memory-bounded official layer load,
offline CPU fallback for Gated-DeltaNet, exact expert-name packing, raw-state
capture, two clean process replays, and no accidental CUDA initialization.

Only after T0-R2 plus independent verification may the real 1.04-GB bank be
built and T0-P2 be run. A T0-P2 pass opens T1 design, not T1 execution.

### Superseding frozen protocol

The independent methodology audit later closed R2/P2 and led to immutable
T0-R4/P4. The current executable is deliberately split. R4-REF-R1 covers the
official BF16 reference, fresh-cache prefix ladder, strict packed-weight
conversion, raw BF16 evidence, and create-new real-Q5 record build/replay. It
cannot emit the final T0-R4 pass. A separate hash-frozen R4-Q5 stage must still
decode/execute the Q5 weights, retain Q5 raw arrays, apply held-out Q5-vs-BF16
metrics and all negative controls. An independent combined verifier must pass
both stages before T0-P4 eligibility opens. This staging prevents an incomplete
bank build from being mislabeled as numerical Q5 evidence.

## Honest next gate after T0

T1 needs all official layers plus final norm/LM head, natural routes at every
layer, held-out vocabulary logits, cross-entropy/top-1 agreement and a
full-depth quality suite. The index declares 40 shards and roughly 159 GB of
checkpoint payload, so T1 must stream shards and must not assume full-model
materialization in RAM. Its downloads, Q5 conversion, physical execution and
claims require a new preregistration and separate authorization.
