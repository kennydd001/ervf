# D10 differentiated-Q5 bank - scientific design audit

**Verdict:** `design_feasible_without_new_large_bank`  
**Execution status:** CPU design verification only; no GPU work or bank write  
**CPU checks:** 10/10 pass

## Scientific problem

D9 is a strong physical capacity-bridge result, but its two correctness oracles
prove different things:

- the differentiated 64-byte headers prove that the staged image contains the
  intended `(layer, expert, projection)` records;
- the Q5 outputs prove arithmetic equality only for an invariant payload,
  because every P0 record uses code bytes `0x55` and BF16 scale word `0x3c00`.

Consequently, D9's wrong-expert and wrong-layer controls change the header
image but cannot change the numerical Q5 output. A compute kernel that silently
read the wrong expert could still reproduce the same output. D10 must join
route identity and numerical output without pretending synthetic canaries are
real checkpoint weights.

## Minimal falsifiable design

Do **not** build another 49.93-GB bank. Keep the existing P0 bank mapped
read-only and reuse D9's 973,209,600-byte active HBM staging plane. For each of
the 480 selected expert records:

1. stage the unmodified record and run D9's full differentiated byte oracle;
2. read the actual layer/expert identity from the staged `SQ5M` header;
3. patch only three BF16 scale words in the staged copy, never the host bank;
4. run the unchanged width-8 Q5/SwiGLU kernels with a frozen one-hot input;
5. compare the three numerical canary outputs against values derived
   independently from the **intended** route table.

The patched locations are UP projection, rows 0, 1 and 2, group 0. Relative to
the expert-record start their byte offsets are:

```
up_base + header + code_bytes + row * groups_per_row * sizeof(BF16)
= 675840 + 64 + 655360 + row * 32
= 1331264 + {0, 32, 64}
```

Define the expert identity and three radix-32 digits as

```
i = 512 * layer + expert                    0 <= i < 24576
d2 = (i >> 10) & 31
d1 = (i >> 5)  & 31
d0 = i & 31
s(d) = BF16(2^-7 * (1 + d/64))
```

All 32 values of `s(d)` are exactly representable in BF16 and stay close to the
original `2^-7` scale. The 3-word tuple is injective over all 48 x 512 routed
experts. It could be materialized in only 147,456 bytes, but should be computed
procedurally.

Use `x[0]=1` and all other input elements zero. The first Q5 code decoded from
the frozen `0x55` payload is `+6`; therefore UP rows 0..2 must contain

```
BF16(6 * s(d2)), BF16(6 * s(d1)), BF16(6 * s(d0))
```

The CPU audit exhaustively verified 24,576 distinct expected output triplets.
A one-word identifier would be invalid: BF16 provides only 128 exact mantissa
states in a fixed binade, fewer than 24,576 identities.

## Required controls and gates

Freeze these before compile or GPU execution:

- positive `all_hot`, `mixed_5_hot_5_cold` and `all_cold_tail` cases;
- same-layer wrong-hot-expert and wrong-hot-layer controls;
- same-layer wrong-cold-expert and wrong-cold-layer controls;
- boundary controls `498 -> 499` and `499 -> 498` so the mapped/cold dispatch
  transition is itself falsifiable;
- the expected canary must be computed from the intended route artifact, while
  the staged canary scales must be computed from the actual staged header;
- no shared pointer table, helper output or candidate header may supply both
  sides of the comparison.

Primary correctness requires:

- zero pre-patch full-image byte mismatches for all positive cases;
- every negative header control detected;
- all 1,440 canary BF16 words per case exactly equal to the intended-route CPU
  oracle and digest-equal;
- every negative pointer substitution causes at least one canary bit mismatch;
- repeated layer/expert IDs produce the same canary and all distinct IDs produce
  distinct triplets;
- finite full outputs and clean 48-range unregister.

Timing must include cold escape, staging, the header-derived patch kernel, Q5
compute and final synchronization. Correctness arrays are small: preserving all
canary words for three positives and six negative controls costs under 26 KiB.
There is no reason to omit them or rely only on mismatch scalars.

## P4D route-capture audit

P4D is genuine model-derived routing evidence, but for the wrong target model:

- local model: `qwen3-30b-a3b-base`;
- 48 layers, 128 routed experts, top-8;
- D9 target geometry: 48 layers, 512 routed experts, top-10;
- five domains x 1,024 tokens, with calibration/validation/test partitions;
- 48 route files totaling 3,964,416 bytes;
- 1,966,080 route IDs, range 0..127, with zero within-row duplicates.

Therefore P4D cannot be called a natural or representative
Qwen3-Coder-Next/512-expert trace. A 128-to-512 remap and two invented extra
experts would be a workload proxy, not target-router evidence. If used, it must
be named `P4D-shaped synthetic proxy`, with mapping learned/frozen only on the
calibration partition and no model-quality or target-frequency claim.

A natural D10 trace requires either:

- a pinned target checkpoint and captured 48 x token x 10 route IDs, or
- a small externally supplied target route artifact with checkpoint revision,
  tokenizer/input hashes, router implementation and capture provenance.

The trace itself is only a few MiB; a second weight bank is not required.

## Strictly required large artifacts

For the numerical D10 mechanism test, the only large artifact required is the
already-existing read-only P0 bank:

- `reports/runs/streamq5_moe/port80b_p0/port80b_p0_full_q5_bank.bin`
  (49,925,652,480 bytes), plus its small manifest.

No new bank, checkpoint clone or differentiated 50-GB copy is justified.
P4D's route files are optional and total about 3.96 MB. They are not needed for
the core numerical wrong-pointer test. A real target checkpoint is required
only if the next claim explicitly includes natural routing or model quality.

## Independent verifier strategy

The eventual D10 verifier must remain CPU-only and should:

1. verify preregistration, runner, manifest, D9 dependency and route hashes;
2. inspect source to prove the bank mapping is read-only and the patch target is
   the HBM staging plane;
3. exhaustively reconstruct the 24,576-entry three-word code and prove
   injectivity after BF16 multiplication/rounding;
4. reconstruct intended route IDs independently and recompute all positive and
   negative canary arrays from stored raw BF16 words;
5. check that observed canary arrays came from the actual-header patch path;
6. recompute inclusive wall statistics and frozen gates from raw samples;
7. verify all controls, error fields and 48 clean unregisters;
8. preserve the claim boundary: synthetic differentiated canary, not real
   weights, natural routing, quality or end-to-end decode.

The design verifier is:

- `scripts/streamq5_moe/verify_d10_differentiated_q5_design.py`
- output: `reports/streamq5_moe/d10_differentiated_q5_design_verification.json`

Current CPU-only result: **10/10 checks pass**. This establishes feasibility of
the falsifiable design, not a D10 experimental pass.
