# S100 phase 5 agent handoff

Run only `RUN_ALL_S100_PHASE5.ps1` from the ZIP root.

Interpretation rules:

- QFAST is the frozen quality-green base: 18.75165 ms / 53.33 tok/s.
- Calibration uses `_01`, validation `_02`, heldout `_03/_04`; never mix them.
- Global K5/K4 and Mamba-W4 are evidence, not adopted components.
- A phase5 candidate is promoted only if heldout status is
  `v18_fidelity_candidate` and fresh timing is valid.
- Do not loosen the original phase-3 heldout gates.
- If selective K is green but small, keep it and move to grouped/downflow work.
- If thresholding is green and materially faster, its next task is a byte/panel
  accounting diagnostic followed by composition with exact-reranked lm_head.
- S100-single is only <=10.000 ms/useful token with the frozen heldout quality
  gate green.
