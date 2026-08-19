# S100 Lightning Phase 15 — provenance reset + BF16x2

Date: 2026-08-19
Target: NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 only.

Nano and Lightning are shape-identical but model-dependent evidence does not
transfer. This phase refuses max_position_embeddings != 1,048,576, quarantines
the inherited trace, creates a fresh Lightning-parent trace, and tests:

- BF16-rounded input through ERVF;
- one-term native BF16 with FP32 output;
- BF16x2 high+residual activation in one GEMM with FP32 output;
- three-term control;
- K/V/O family-selective variants.

Cold timing streams all BF16 matrices once per repetition. Candidate selection
uses calibration only; heldout is read only after strict validation passes.

Final flags:
- LIGHTNING_TRACE_PROVENANCE_GREEN
- BF16X2_COLD_STREAM_OPEN
- BF16X2_QUALITY_OPEN
- BF16X2_FAMILY_SELECTIVE_OPEN
- LIGHTNING_BLOCK_VERIFIER_RERUN_OPEN
