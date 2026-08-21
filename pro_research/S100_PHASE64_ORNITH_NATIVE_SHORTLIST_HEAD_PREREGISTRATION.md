# S100 Phase64: native shortlist + exact ERVF rerank preregistration

## Motivation

Phase63 showed that a direct ERVF M4 head remains 4.91 ms. The same real head's
native Blackwell NVFP4 matrix path is near 1.29 ms, but quantizing activations to
NVFP4 can alter logits. This phase uses native logits only as a shortlist, then
recomputes shortlisted rows with the original F32 activation and exact ERVF
reduction.

## Frozen design

- Real Pottokao 248,320 x 2,048 NVFP4 LM head.
- 32 deterministic random finite rows, evaluated as eight H4 blocks.
- One static activation tensor scale calibrated from all 32 rows with 10%
  amax margin.
- Native `scaled_mm` produces BF16 logits and top-64 IDs per position.
- A custom indexed ERVF kernel recomputes those 64 logits from natural
  checkpoint bytes and original F32 inputs; argmax is taken only after rerank.
- Exact control is four production ERVF H1 heads.
- Timed candidate includes fused activation quantization, native head, top-64,
  exact rerank and final gather/argmax. Weight/scale repacking is setup work.

## Gates

1. Native top-64 contains the exact control top-1 for all 32 rows.
2. Exact rerank returns the control token ID for all 32 rows and repeats exactly.
3. Indexed ERVF shortlist scores are bit-identical to the corresponding full
   ERVF logits.
4. Candidate H4 median is below 2.0 ms and at least 2.5x faster than H1 x4.
5. Indexed ERVF and quantizer kernels use no local memory and at most 64
   registers/thread.

Passing is a synthetic-activation component result. Real final-normalized
Ornith activations still require end-to-end adjudication.
