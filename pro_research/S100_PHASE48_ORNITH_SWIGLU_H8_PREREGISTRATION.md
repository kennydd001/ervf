# S100 Phase48 — real Ornith NVFP4 SwiGLU H8 expert

## Question

Does the Phase33 exact NVFP4 M8 kernel transfer from Nemotron projections to
one complete, checkpoint-real Ornith-1.5 routed expert: gate projection, up
projection, SiLU multiplication and down projection?

## Frozen input

- Checkpoint: pinned snapshot
  `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash@d60d98b0b2feeabca19196005f4ac378279e2f25`
- Tensor: expert 0 in layer 20.
- Shapes: gate/up `[512, 2048]`, down `[2048, 512]` after unpacking.
- Batch: eight deterministic, distinct FP32 hidden rows.
- GPU: NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU (SM120, 8 GiB).
- H8 implementation: Phase33 `NVFP4M8Warp32`; H1 control: production
  `FusedNVFP4` ERVF GEMV; both consume the checkpoint's packed E2M1 codes,
  group-16 E4M3 scales and F32 global scales without materialising weights on
  GPU.

## Measurements

1. Full-output comparison against an independent NumPy byte-level NVFP4 decode.
2. Full-output comparison against eight independent H1 expert evaluations.
3. Hot-resident latency for one H1 expert, eight sequential H1 experts and one
   H8 expert.
4. Pinned-host-to-device staging plus H8 latency for the complete 1.6875 MiB
   expert record.
5. The padded-H8 break-even route multiplicity derived from measured H1 and H8
   latency.

## Gates

1. All nine checkpoint tensors have the frozen shapes and dtypes.
2. Independent decoded reference and both GPU paths are finite.
3. H8 versus independent-reference normalized RMSE is <= 0.005, cosine is >=
   0.9999 and normalized max error is <= 0.020.
4. H8 versus eight-H1 normalized RMSE is <= 0.001 and normalized max error is
   <= 0.005.
5. Repeated H8 execution is bitwise deterministic.
6. Hot-resident complete H8 expert latency is <= 0.50 ms.
7. H8 is at least 4x faster than eight sequential H1 expert evaluations.
8. Pinned staging plus complete H8 execution is <= 1.50 ms.
9. Padded H8 breaks even at route multiplicity <= 4.

## Claim boundary

Passing proves the complete SwiGLU expert math and the maximum-reuse H8 kernel
geometry for one real expert. It does not prove that eight real target tokens
choose the same experts. A target-decoder route census is required to turn the
break-even multiplicity into a whole-model throughput claim. It also excludes
linear attention, full attention, routing weights, shared-expert gating,
DFlash acceptance and token sampling.

