# S100 Phase 15 preregistration

Checkpoint: current cached Nemotron 3.5 Lightning checkpoint used by Phase 14.

## A — component variants

B in {1,4}. Every BF16 case from Phase14 D2.

Variants:
- `mm_bf16out`
- `mm_fp32out`
- `mm_fp32out_comp2`

Cold cache scrub >= 4x L2 before each timed sample.

Component research-go:
- B1 speedup >= 1.10x for `mm_fp32out`, OR
- B4 speedup >= 1.25x for `mm_fp32out_comp2`;
- max NRMSE reported, never hidden.

## B — corrected teacher-forced fidelity

The exact parent generates the greedy target chain.
The candidate gets:
- the identical prompt token ids;
- after each scored position, exactly the exact parent's target token.

It must never advance with its own greedy token in this test.

Calibration/validation only until one arm is locked.

Arms:
- mm_fp32out / all
- mm_fp32out / attention
- mm_fp32out / mamba
- mm_fp32out_comp2 / all

Strict validation gate:
- top1 >= 0.970
- top5 >= 0.999
- mean CE delta <= 0.025
- mean coarse KL <= 0.015
- p95 coarse KL <= 0.060
- finite

## C — matrix sensitivity

If whole-stack arms are not strict-green, each live BF16 matrix is substituted
alone with mm_fp32out on calibration only. Rank by fidelity damage and Phase14
B1/B4 component saving.

A matrix is "locally safe" only when:
- top1 >= 0.995
- top5 = 1.000
- mean CE delta <= 0.005
- mean KL <= 0.003

This is selection data, not heldout evidence.

## D — exact-state draft horizons

At every block boundary:
1. snapshot exact recurrent state;
2. run exact parent H steps greedily;
3. restore exact pre-state;
4. run native candidate H steps greedily from the same first token;
5. compare accepted exact prefix;
6. restore exact post-state before the next block.

H in {1,2,4,8}. Validation prompts only.

Block research-go at H4:
- first-token agreement >= 0.95
- mean accepted exact prefix >= 1.5 tokens
- full H4 match rate >= 0.25

This is not a throughput claim.

## E — heldout

Heldout `_03/_04` is only evaluated after a single variant/scope is locked by
validation. Same exact-parent direct trace protocol.

Heldout must satisfy the Phase15 strict gate above before a runtime build may
open.
