# S100 Phase 13A — lossless entropy census

Date: 2026-08-18

## Verdict

**CLOSED at discovery.** The lossless palette/escape estimate does not open
the Phase 13A gate on the exact checkpoint used by Phase 12C.

- 140 resident matrices were covered, plus 18 deterministic routed-expert
  samples from layers 1, 27 and 51.
- Mamba FP8 best estimate: **8.000 bits/weight**, versus the preregistered
  `<=6.0` gate.
- All resident matrices best estimate: **96.801%** of raw bytes, versus the
  preregistered `<=70%` gate.
- Independent verifier: **PASS**; promotion: **false**.

The estimate includes an uncompressed raw-byte fallback, so it cannot claim a
compression ratio worse than the source representation. It is still only an
encoding estimate: no GPU decoder, scale-plane metadata accounting, latency,
or quality run was performed.

## Checkpoint identity

The census uses `models/nemotron_3_5_lightning`, the same five-shard checkpoint
used by Phase 12C. The `_v35` directory is a different checkpoint and was not
mixed into this result.

The large `lm_head` matrix was analyzed using every 16th row through the
safetensors slice API; its full byte and weight totals remain in the aggregate
accounting and the sampling is marked in the JSON artifact.

## Interpretation

Lossless entropy coding is not a credible primary path to the missing 100
tok/s factor for this checkpoint. It remains potentially composable as a
secondary transport optimization, but no entropy decoder should be built under
the frozen promotion order unless a new measured premise is supplied.
