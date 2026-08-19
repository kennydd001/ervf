# S100 Lightning Phase 16 — handoff

Phase 15 established a fresh, exact Lightning trace and a 19.28 ms/token parent.
Its cold Tensor-Core results were real, but native K/V/O quality was poor.
The strongest concrete implementation hypothesis is a producer/consumer stream
handoff error: Phase 15 called `torch.from_dlpack` before entering the Torch
ExternalStream wrapping the CuPy producer stream. O is the freshest input and
the most destructive family.

Phase 16 does not assume that hypothesis is true. It compares:
- legacy ordering;
- context-first DLPack conversion;
- an explicit producer-sync control.

If the hypothesis is false, real-input shadow arithmetic and one-matrix
substitution still tell us whether the remaining divergence is ordinary
recurrent amplification or a layer/family-specific defect.

All model-dependent Phase-12/14 DFlash2 results are rerun on Lightning.
