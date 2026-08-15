# STREAMQ4-MoE P0 - full-depth quality preregistration

Locked on 2026-08-12 before any STREAMQ4 validation or test output was opened.

## Hypothesis

The CORETAIL failure is primarily a low-bit quality failure, not proof that
expert streaming is too slow. Replacing the resident Q2 expert bank with a
host-resident symmetric Q4 bank and replacing the naive INT4 trunk with INT8
should preserve full-depth model quality while leaving enough GPU memory for a
large route-conditioned expert cache.

The primary candidate is `Q4 experts + INT8 trunk`. Q4 and INT8 use symmetric
per-row group-128 quantization, round-to-nearest-even, codes `[-7,7]` and
`[-127,127]`, BF16 scales, and immediate BF16 dequantization in this quality
screen. RMSNorm vectors remain BF16. No clipping, group-size, layer, codebook,
or mixed-precision sweep is allowed.

## Fresh data lock

Five domains are fixed: general, code, math, multilingual, and instruction.
Validation and test each contain two 128-token contexts per domain. These
contexts must be disjoint from each other and from the earlier CORETAIL P2
validation/test artifact. Their source paths, offsets, hashes, tensor hashes,
and model hash are locked before the first forward pass.

The previous CORETAIL P2 outputs may motivate this new architecture but may
not be reused as a STREAMQ4 decision set. STREAMQ4 test opens only once after
the validation progression decision is stored.

## Full-depth variants

1. BF16 teacher;
2. Q4 experts + BF16 trunk - expert isolation;
3. BF16 experts + INT8 trunk - trunk isolation;
4. Q4 experts + INT8 trunk - primary candidate;
5. Q4 experts + INT4 trunk - negative/control diagnostic.

All 48 decoder layers, embeddings, LM head, router, and non-expert matrices are
included according to the variant definition. Metrics are next-token
cross-entropy, relative CE versus BF16, top-1 agreement, per-domain metrics,
per-layer/final hidden relative-L2 and max-abs, runtime, CUDA peak, and RSS.

## Gates

- Validation progression: primary relative CE `<=3%`, all finite, all 48
  layers, all input/provenance hashes correct. Otherwise close without test.
- Quality pass: primary relative CE `<=2%` on both validation and the once-only
  fresh test split, with all controls valid.
- Test `>2%` closes the fixed Q4/INT8 quality candidate. There is no repair or
  configuration sweep in this registry.
- A quality pass opens physical Q4-bank construction, exact byte accounting,
  cache simulation, Q4 microkernel benchmarking, and finally an integrated
  wall-clock screen. It does not itself prove tokens per second.

## Memory and performance claim boundary

The Q4 bank resides in pinned host memory or storage, not permanently on the
GPU. GPU accounting must include the INT8 trunk, 4K BF16 KV cache, 0.75-GiB
runtime reserve, cache records, indices, padding, and staging buffers. No
roofline projection may be called a wall-clock result.
