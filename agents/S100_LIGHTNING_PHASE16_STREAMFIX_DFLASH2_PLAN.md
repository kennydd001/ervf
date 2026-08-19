# S100 Lightning Phase 16 — frozen plan

Target: Nemotron 3.5 Lightning only.

1. Adjudicate the DLPack/ExternalStream handoff with legacy, context-first and
   synchronized controls on asynchronous producers and real decode activations.
2. Screen every K/V/O matrix independently under TC1 and TC2.
3. Build calibration-only greedy safe subsets; validation and heldout remain
   untouched until their gates open.
4. Re-run the perfect-draft block verifier and route-union census on Lightning.
5. Re-run DFlash2 suffix correction, candidate-lattice selection and resident
   memory economics on Lightning.
6. Do not reuse Nano model-dependent evidence.
