# STREAMQ5-MoE P0 - full-depth quality preregistration

Locked on 2026-08-12 after STREAMQ4 validation closed and before any STREAMQ5
validation or test output was opened.

## Hypothesis

A host-resident symmetric Q5 expert bank with an INT8 trunk preserves
full-depth quality within 2% while retaining a plausible local systems path:
Q5 increases raw expert transfer by only 25% over Q4 but doubles the positive
code resolution from 7 to 15.

The primary candidate is `Q5 experts + INT8 trunk`. Q5 uses symmetric per-row
group-128 quantization, round-to-nearest-even, codes `[-15,15]`, and BF16
scales. INT8 uses the same fixed rule with codes `[-127,127]`. Immediate BF16
dequantization is used for this quality screen; RMSNorm vectors remain BF16.
No clipping, group-size, layer, mixed-precision, or codebook sweep is allowed.

## Fresh split and variants

Validation and once-only test each contain two new 128-token contexts in each
of general, code, math, multilingual, and instruction. They must be exact-
context disjunct from both CORETAIL P2 and STREAMQ4 P0. Source hashes, offsets,
tensor hashes, model hash, evaluator hash, and all decisions are locked before
the first forward pass.

Full-depth variants:

1. BF16 teacher;
2. Q5 experts + BF16 trunk;
3. BF16 experts + INT8 trunk;
4. Q5 experts + INT8 trunk - primary candidate;
5. Q5 experts + INT4 trunk - diagnostic control.

## Gates

- Validation progression: primary relative CE `<=2.5%`, all values finite,
  all 48 layers, and all provenance controls valid. Otherwise close without
  test.
- Quality pass: validation and test primary relative CE both `<=2%`.
- Any test result above 2% closes this fixed candidate. No repair is allowed.
- Passing quality opens exact Q5 byte accounting, physical bank construction,
  route-cache simulation, Q5 microkernel/transfer benchmarking, and only then
  an integrated wall-clock screen.

P0 proves quality only. No cache projection or microbenchmark is an end-to-end
tokens-per-second result.
