# Qwen GPTQ Bank — P0 full-bank preregistration

Locked on 2026-08-11 before opening the supplemental routes.

## Objective

Build a complete canonical GPTQ-2bit bank for all 48 × 128 Qwen3-30B-A3B experts. This is a source-asset campaign for the previously blocked physical CORETAIL full-bank P0; it does not reopen any closed E2GQ, FLEQ, HERA, or DHERA claim.

## Immutable model and quantizer

- Model: local `Qwen3-30B-A3B-Base`, revision `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
- Official architecture: 48 layers, 128 experts/layer, top-8, hidden 2,048, expert intermediate 768.
- Pinned GSQ commit: `03fc16484c369e3127225615d5e03e8d3a6043e3`.
- Quantizer: the pinned `src/prior/gptq.py` plus `src/prior/quant.py` path already used by FLEQ.
- Quantization: symmetric 2-bit, per-output-channel MSE grid, group size 128, `percdamp=0.1`, block size 128, `static_groups=False`.
- Code alphabet: `{-2,-1,0,1}`; scales are stored as raw BF16 bits.
- RTN substitution is forbidden.

## Calibration data and row rule

The calibration universe is the ordered union of the immutable HERA train-domain lock, immutable DHERA validation-domain lock, and the new supplemental input lock. Existing route results were used only to size the supplement. Supplemental token IDs, offsets, source hashes, ordering, and context boundaries must be locked before supplemental routing starts.

For every `(layer, expert)`, take the first 128 true official top-8 routed rows in this fixed order:

1. HERA domains in `general, code, math, multilingual, instruction` order;
2. DHERA domains in that same order;
3. supplemental domains in that same order;
4. within a domain: context-major, token-major, top-k-slot-major order.

Gate and up GPTQ use the corresponding original BF16 MoE input. Down GPTQ uses `silu(gate(x)) * up(x)` recomputed from the same original BF16 expert weights and selected input row. No duplicated or synthetic activation row may count twice.

## Gates

1. Coverage: every one of 6,144 layer-expert pairs has at least 128 true routed rows.
2. Capture: every saved expert calibration tensor is exactly `[128, 2048]` BF16 and finite; its route provenance is recorded.
3. Quantizer equivalence: the accelerated batched implementation must reproduce the pinned official routine's integer codes and BF16 scale bits exactly on a locked cross-layer/expert/matrix audit set. Failure forces fallback to the official per-matrix implementation; semantics may not be relaxed.
4. Bank completeness: 6,144 experts and 18,432 matrices are physically present with no duplicate identity.
5. Code/scale validity: all codes are in `{-2,-1,0,1}`, every scale is finite/nonzero, and packed-code round trips are exact.
6. Source integrity: every layer artifact, manifest, source tensor, calibration tensor, and pinned implementation is hashed.
7. Independent verifier: all counts, identities, hashes, packed-code decodes, code histograms, scale bits, and size arithmetic are recomputed without trusting the producer summary.

Only after all seven gates pass may CORETAIL `P0_FULL_BANK_ACTUAL_FORMAT` move from blocked to open. Runtime or quality claims remain closed until their own preregistered measurements.

## Resources and resumability

- CUDA allocation ceiling: 7.5 GiB.
- Process RSS ceiling: 32 GiB during routing/capture and 40 GiB during layer assembly.
- Work is append-only and resumable at layer boundaries. Existing completed artifacts are verified and skipped; never silently overwritten.
- A failed or interrupted attempt is preserved and reported.

## Claim boundary

Passing this campaign proves a complete, reproducible GPTQ source bank exists. It does not itself prove CORETAIL fits, improves quality, or accelerates inference.
