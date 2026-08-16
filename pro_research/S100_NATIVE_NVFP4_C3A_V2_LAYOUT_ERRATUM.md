# C3A-v2 — NVFP4 scale-layout implementation erratum

Date: 2026-08-16  
Failed implementation commit: `8d922ac50c3ccc6c45777af21d690c60df3b9536`  
Branch: `pro-s100-nativefp4-c2b`

## What the first C3A run established

The first real-checkpoint run did **not** pass representational correctness. All four
real-weight native calls were finite and M=8/M=1 remained <=1.03, with honest >=4x-L2
rotation, but the independent dequant reference gates failed. Checkpoint hashes and
reference recomputation were independently valid. Therefore the failure is treated as
an implementation/ABI mismatch, not model-quality evidence and not a performance win.

## Root cause

C1 and C3A-v1 used a custom self-invertible `SWIZZLE_32_4_4` mapping whose outer
128-row x 4-scale-column tiles were emitted in K-block-major order:

`outer = k_block * n_row_blocks + row_block`

TorchAO's reference `to_blocked()` uses:

`padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)`

which flattens the outer tiles row-block-major:

`outer = row_block * n_col_blocks + k_block`

The inner 32x16 interleave was already consistent. C1 proved that its own permutation
round-tripped losslessly; it did not independently prove that the outer block order was
the native ABI. Synthetic all-one scale tests were invariant to this mistake.

## C3A-v2 correction

C3A-v2 leaves the failed v1 source intact and applies an additive in-process correction:

1. checkpoint `[N,K/16]` scale bytes are padded and rearranged by a direct mirror of
   TorchAO `to_blocked()`;
2. a non-uniform 256x8 byte witness covers 2 row blocks x 2 column blocks and must match
   an independent row-block-major closed form with zero byte mismatches;
3. the same witness must differ from the legacy K-major mapping, proving discrimination;
4. a native two-level FP4 smoke uses four distinct 128x4 E4M3 scale tiles (0.5, 1, 2, 4).
   Its exact BF16 expected outputs are 48 for the first 128 output rows and 192 for the
   second 128. C3A-v1's block order cannot satisfy this test;
5. only after those preflight gates pass is the original frozen C3A real-checkpoint
   diagnostic rerun with the corrected repacker.

The independent C3A-v2 verifier does not import the layout implementation; it rebuilds
the witness with stdlib indexing. The original C3A independent checkpoint/reference
verifier then runs unchanged.

## Threshold policy

No C3A numerical or performance threshold is changed:

- normalized RMSE <= 0.020;
- cosine >= 0.9990;
- normalized max absolute error <= 0.050;
- identical M8 row normalized max difference <= 0.005;
- cold working set >=4x L2;
- M8/M1 p50 <=1.15 for at least 3/4 families, including lm_head if measured.

A green v2 run would validate only real checkpoint B representation under exact +1 A and
the measured M geometry. Real activation NVFP4 quality remains C3B; grouped MoE and full
model integration remain later gates.
