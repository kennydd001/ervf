# S100 Lightning Phase 18R — minimal dual-graph repair

Date: 2026-08-20

Phase 18 produced no valid MoE surgical timing result. The exact MoE output table was eventually recorded with 10/10 parent parity, but every oracle arm still failed parity before timing. The last untested local patch proposed two CUDA graphs on one runtime.

## Frozen repair

Phase 18R starts from a fresh `s100_phase10a_runtime.build()` parent and never restores semantic state from snapshots.

For every prompt:

1. `_reset_exact_state(rt)`;
2. select the untouched production parent graph;
3. run the whole prompt so routed-expert cache/LRU, hidden, Mamba, convolution and KV state are canonical;
4. set one per-prompt GPU table offset outside the timed region;
5. switch only `rt._graph` to the already captured record/overhead/oracle graph;
6. run target tokens.

Replay-table indexing is device-derived inside the graph:

`table_index = *rt._pos_dev + *table_offset_dev`

There is no per-token host->device replay-index transfer.

## Stage gates

R0: exact parent baseline manifest.

R1: record exact full-MoE outputs with `original_moe(out)` followed only by a device record-copy. Require 10/10 generated IDs and final semantic fingerprints equal to R0 and a completely finite replay table.

R2: full-MoE E oracle parity smoke. Prompt uses the stored parent graph; targets use the E graph. Require 10/10 parity before any performance run.

R3: E A/overhead/oracle1/oracle2/B timing. Overhead is the exact parent graph plus the same replay-copy into scratch. It is not a plain parent control.

R4: only if R2/R3 are green, run surgical D/PD/UPD/RD/S/EMPTY_E/interactions and per-layer E_L* arms with the same dual-graph prompt protocol.

## Parity

Hard semantic fingerprint:

- generated token IDs;
- final logits SHA256;
- hidden vector;
- Mamba SSM + convolution state;
- used FP8 KV bytes;
- device position;
- finite logits.

Expert-cache metadata is deliberately not a semantic parity item for oracle targets, but prompt construction is always performed by the parent graph so partial surgical arms start from the canonical cache state.

## Claim boundary

No Phase-18 timing survives unless E parity is green under this protocol. Phase-17 remains the authoritative MoE ceiling (~10.68 ms corrected saving) until Phase 18R reproduces it.
