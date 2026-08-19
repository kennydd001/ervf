# S100 Phase 14 agent handoff — DFlash2 revision

## Frozen target

`models/nemotron_3_5_lightning`, current QFAST + `alpha=0.0003` quality parent,
8 GiB VRAM and single-stream S100.

## Current state

- 13A entropy: closed for the tested lossless hypothesis.
- 13C temporal delta: closed for the tested energy gate.
- 13D native BF16: primary survivor; 5.47x B=4 component result, model-level
  validation still required.
- 13B: incomplete rather than falsified; previous screen omitted output error,
  token fidelity and runtime.
- 13E: raw-code rank-SVD not promotable; one decoded activation-weighted kill
  screen remains justified.

## DFlash2 verdict

DFlash2 is relevant but is not a drop-in kernel. It requires a trained,
model-specific drafter plus grouped dynamic two-tap convolution, candidate
codebooks, hidden projection and lossless verifier support.

Do not load the public Qwen sidecar into Nemotron. Do not train a full draft
until the current block verifier has positive S100 budget.

Phase 12A perfect-draft block verification is approximately 35.6/71.0/143.9 ms
at B=2/4/8, which caps the current path near 56 tok/s even with zero draft cost.
Phase 12C's optimistic B=4 projection is about 60.9 ms, or 65.6 tok/s. DFlash2
acceptance cannot repair this verifier ceiling.

## Phase 14 work

- 14D: broaden native BF16, then strict validation and gated heldout.
- 14B2: real X->Y reduced-rank output reconstruction.
- 14E2: decoded NVFP4 expert basis with activation-weighted residuals.
- 14F0: verifier economics plus actual resident-memory envelope.
- 14F1: target-only transfer proxy for suffix correction and candidate-lattice
  path-selection headroom.

## Final DFlash2 flags

```text
DFLASH2_CURRENT_VERIFIER_S100_OPEN
DFLASH2_RESIDENT_MEMORY_OPEN_4K
DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN
DFLASH2_TRAINING_BUILD_OPEN
```

The last flag only authorizes a later draft-training phase. It cannot claim a
runtime speedup or S100.

Read `agents/S100_PHASE14_FULL_RESEARCH_REPORT.md` before changing gates or
claim boundaries.
