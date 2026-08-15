# Qwen GPTQ Bank — P0 supplement B preregistration

Locked on 2026-08-11 before opening supplement-B routes.

Supplement A was deliberately allowed to finish as an immutable coverage attempt. Its already-opened route artifacts show two residual deficits: layer 13/expert 99 has 89 rows and layer 43/expert 95 has 113 rows. The original 128-row gate is unchanged.

## Locked remediation

- Instruction: the next 524,288 tokens from the pinned Dolly source, starting at token offset 196,608.
- Math: the next 262,144 tokens from the pinned GSM8K training source, starting at token offset 163,840.
- Context boundaries remain 1,024 tokens; source text construction and tokenizer are identical to supplement A.
- Both token tensors, their source offsets, source hashes, and tensor hashes must be physically locked before routing.
- Official top-8 routes are captured for every layer for both tensors. No source or size may be changed after any supplement-B route is opened.

## Gate and boundary

Coverage passes only if the cumulative immutable HERA + DHERA + supplement A + supplement B count is at least 128 for every one of 6,144 layer-expert pairs. A further failure is preserved and requires another separately preregistered attempt. This addendum changes neither GPTQ settings nor any downstream gate.
