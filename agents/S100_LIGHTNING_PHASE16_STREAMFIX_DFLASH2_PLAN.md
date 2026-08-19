# S100 Lightning Phase 16 — native stream adjudication, selective attention, and DFlash2 reset

Date: 2026-08-19
Parent: `agent/s100-lightning-phase15-hardware@ccc5f81`
Target: NVIDIA Nemotron 3.5 Lightning only.

## Evidence entering Phase 16

- Fresh Lightning trace is exact against its parent.
- Cold B=4 BF16 Tensor-Core speed is real but terms=1 is only 2.48x and terms=2 is 1.46x across the complete 8.37x-L2 stream.
- Component arithmetic for BF16x2 is accurate, yet end-to-end K/V/O substitutions diverge badly.
- O is substantially more destructive than K or V.
- Nano-derived verifier, route-union, hidden-state and DFlash2 results are quarantined for Lightning.

## Primary bug hypothesis

Phase 15 converts CuPy arrays with `torch.from_dlpack` before entering the Torch `ExternalStream` wrapping the CuPy producer stream. DLPack synchronization may therefore be established for Torch's previous current stream while the copy/GEMM is launched on another stream. The O input is produced immediately before O-projection and is the most race-sensitive; isolated component inputs were already materialized and synchronized.

This hypothesis is falsified or confirmed with three paths on the same real activations:

1. `legacy`: Phase-15 ordering;
2. `context_first`: enter the external stream before all DLPack conversions;
3. `sync_control`: explicitly synchronize the CuPy producer before the legacy call.

## Phase 16A — producer/consumer sentinel and shadow arithmetic

- Generate inputs asynchronously on the exact CuPy stream.
- Run legacy/context-first/sync-control native K/V/O.
- Compare each to current ERVF.
- During real teacher-forced Lightning execution, compute native outputs into scratch while ERVF remains authoritative.
- Record per layer, family and token: input fingerprint, output NRMSE/cosine/max-abs, stream pointers, first outlier, and repeat determinism.

A stream bug is confirmed when context-first and/or sync-control removes an error absent from the isolated synchronized component test.

## Phase 16B — one-matrix and cumulative substitution

For each of the six attention layers and K/V/O:

- substitute exactly one matrix using fixed TC1 and TC2;
- run a frozen 16-token-per-prompt calibration screen;
- record top-1/top-5/CE/KL and first divergence.

Then evaluate greedily built safe sets:

- per family;
- per layer;
- combinations selected only from calibration;
- full strict validation and heldout only after passing.

A candidate must also beat the matched parent by >=0.15 ms/token in fresh A/C/C/B timing to promote.

## Phase 16C — Lightning block verifier reset

Only a quality-green native subset may enter the block runtime. Re-run on Lightning:

- perfect-draft B={2,4,8} correctness and complete cycle cost;
- route-union/expert-row census;
- state/KV shadow commit;
- no Nano result or projection is reused.

`LIGHTNING_PERFECT_DRAFT_S100_OPEN=true` requires the measured complete verifier to exceed 100 useful tok/s before draft cost.

## Phase 16D — DFlash2 Lightning transfer screen

Re-run the suffix-decay and candidate-lattice ideas on fresh Lightning hidden states and labels. Recompute resident memory against the actual Lightning parent and use only measured Lightning verifier economics.

Training opens only if all are true:

- measured Lightning verifier leaves positive S100 draft budget at a realistic acceptance length;
- a resident drafter configuration fits with reserve;
- suffix correction or lattice selection produces preregistered validation signal.

## Decisions

- `STREAM_HANDSHAKE_BUG_CONFIRMED`
- `FIXED_NATIVE_SHADOW_GREEN`
- `FAMILY_SELECTIVE_NATIVE_PROMOTE`
- `LIGHTNING_PERFECT_DRAFT_S100_OPEN`
- `DFLASH2_LIGHTNING_SIGNAL_OPEN`
- `DFLASH2_TRAINING_BUILD_OPEN`

Technical failures remain null. Component and proxy results cannot claim S100.