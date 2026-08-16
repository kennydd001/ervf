# S100 DualRHS-ERVF preregistration

Date: 2026-08-16
Branch: `pro-s100-dualrhs`
Base: `pro-s100-k2-oracle-v18@d29e5cc565c5e48715bc194db2c70a16bd4b3754`
Frozen before any DualRHS timing.

## Why this experiment exists

The V18 K=2 correctness-first layer-major oracle is numerically successful but physically almost identical to two sequential target steps. The full local run reported:

- sequential midpoint: `39.153525 ms / 2 tokens`;
- layer-major K2: `38.67655 ms / 2 tokens`;
- exact state/token/continuation/control gates: all PASS;
- speedup: only `1.01233x`;
- S100 feasibility gate: FAIL (`19.285 ms` required).

That result does **not** show that two-position verification cannot amortize target work. The current K2 body still invokes almost every weight-streaming projection once per position. It changes schedule order, not the amount of common weight traffic.

Two consecutive positions entering a layer share the same projection weights. A memory-bound target can in principle load/decode each weight scalar once and apply it to two independent activation vectors while preserving each position's original FMA stream and reduction tree.

Prior generic MRHS kernels do not settle this question. Their N=2 implementation held all virtual accumulators for both RHS values live in registers. It produced several strong component signals (especially small K/V) but regressed on large NVFP4 families and the LM head. This new candidate changes the physical register geometry while preserving the same logical arithmetic.

## Hypothesis

**Streamed Virtual Accumulator DualRHS-ERVF** can reuse common weight bytes across exactly two RHS vectors without the register-pressure failure of the earlier generic MRHS kernels.

For each output row and each reference virtual thread `tid`, the candidate:

1. loads/decodes a weight once;
2. performs the reference FMA into RHS-0's accumulator;
3. performs the same weight's reference FMA into RHS-1's accumulator;
4. finishes that virtual thread's complete stride-256 MAC stream;
5. stores the two finished virtual accumulators to shared memory;
6. moves to the next virtual thread;
7. after all 256 logical virtual accumulators exist, reconstructs the exact production reduction tree independently for RHS 0 and RHS 1.

The key difference from old MRHS is that the main MAC loop keeps only **two live accumulators at a time**, not `NRHS × virtual_threads_per_lane` accumulators. Shared memory holds completed logical accumulator leaves until exact reduction.

This is not Tensor Core arithmetic, activation quantization or approximate inference.

## Candidate geometry

Frozen candidate:

- two RHS only;
- logical reference width = 256 virtual threads/output row;
- physical ERVF width = 16 lanes/output row;
- 6 output rows / 96 physical threads per block;
- each physical lane sequentially emulates 16 reference virtual tids;
- two activation vectors are staged once per block in shared memory;
- completed virtual accumulator leaves are staged as `[row_in_block, rhs, 256]` FP32;
- exact reference reduction tree is reconstructed with the already-proven ERVF width-16 mapping.

`6 rows/block` is frozen because it keeps dynamic activation+leaf shared memory below the ordinary 48 KiB region for the largest registered `cols=4096` shape, avoiding an opt-in large-shared-memory confound in the first test.

## Registered real checkpoint families

The microbenchmark must use real loaded Lightning weights and cover:

1. attention Q BF16;
2. attention K BF16;
3. attention V BF16;
4. attention O BF16;
5. Mamba in-proj FP8-per-tensor;
6. Mamba out-proj FP8-per-tensor;
7. MoE router F32;
8. shared-expert up NVFP4 + ReLU^2;
9. shared-expert down NVFP4;
10. LM head NVFP4.

Reference performance is the **adopted V18 exact single-RHS dispatch**, called twice:

- selective DenseERVF for the shapes V18 already accelerates;
- production kernels for shapes V18 deliberately leaves alone;
- adopted `FusedNVFP4` ERVF for NVFP4.

Do not compare the candidate to obsolete pre-ERVF kernels.

## Correctness gates

For every registered family and three independent full-mode activation batches:

- `G1_ref_rhs0_bitexact`: candidate RHS0 bits equal adopted single-RHS reference;
- `G2_ref_rhs1_bitexact`: candidate RHS1 bits equal adopted single-RHS reference;
- `G3_candidate_deterministic`: repeat candidate output bit-identical;
- `G4_no_nan_inf`;
- `G5_all_10_families_present`.

Any correctness failure closes this candidate. No tolerance or token-only fallback is allowed under this preregistration.

## Timing protocol

CUDA-event ABBA per family after compile/warmup:

- REF_A: two adopted single-RHS calls;
- CAND_A: one DualRHS call;
- CAND_B;
- REF_B.

Smoke: 6 repeats × 2 ABBA rounds.
Full: 40 repeats × 6 ABBA rounds, three correctness activation batches.

Report per family:

- reference pair ms;
- candidate pair ms;
- speedup;
- reference drift;
- physical weight bytes;
- effective reference/candidate weight-GB/s counting each matrix's bytes once for the candidate and twice for the reference only as an accounting diagnostic.

Also weight each family by its calls per normal target token and report the projected common-projection saving per K2 block.

## Performance gates

The current measured K2 block anchor is `38.67655 ms`; S100 requires `<19.285 ms` before real draft cost.

This microkernel alone is not expected to solve recurrent/attention/MoE-routing work, therefore its opening gates are deliberately substantial but not equivalent to S100:

- `P1_weighted_common_projection_speedup_ge_1_50x`;
- `P2_projected_common_projection_saving_ge_6_0ms_per_K2_block`;
- `P3_mamba_in_speedup_ge_1_40x`;
- `P4_lm_head_speedup_ge_1_35x`;
- `P5_no_registered_family_below_0_90x`;
- `D1_max_reference_drift_le_0_07` in full mode.

**Integration opens only if all correctness gates and P1/P2/P5/D1 pass.** P3/P4 are diagnostic strong-family gates: if one fails but total P1/P2 passes, integration may still open because the measured weighted sum is authoritative.

If projected saving is `<3 ms`, close this physical geometry without post-hoc row/block/register sweeps. A materially different design (Tensor Core, persistent weight tile, routed-expert union) must receive a new registry/preregistration.

## Claim boundary

A micro pass proves only that exact two-RHS common-weight reuse is physically worthwhile on registered Lightning projection shapes. It does not claim a full K2 verifier speed, MTP acceptance, 100 tok/s, or user-visible generation speed.
