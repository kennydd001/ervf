# S100 Phase 15 — agent handoff

## Corrected interpretation of Phase 14

D2 is positive hardware evidence:
- B1 1.307x
- B4 2.814x
- real checkpoint BF16 matrices
- cold cache scrub
- max NRMSE about 0.003

K2 is NOT valid evidence for "same teacher-forced prefixes":
`candidate_against_exact()` advances with `rt.step(int(nxt))`, where `nxt`
comes from the native candidate itself. It does not feed
`ref["targets"][step]`. Its 0.625% top1 and K16=10% therefore conflate numerical
state drift with token-prefix divergence.

K2 also uses torch.mv, while the measured D2 primitive uses torch.mm with a
transposed contiguous matrix.

## Phase 15 research questions

1. Does D2's actual MM primitive retain the B1/B4 speedup with FP32 output?
2. Does FP32 output materially reduce causal drift?
3. Is residual-compensated BF16 input accurate enough for a B4 verifier/draft?
4. If "all BF16" is not safe, which individual matrices/families are?
5. If exact recurrent state is refreshed at every block boundary, how many
   native draft tokens remain exact for H=1/2/4/8?

## Claim boundary

- component = hardware primitive only;
- teacher forced = same input tokens, candidate state may drift;
- exact-state horizon = native draft starts from exact recurrent state;
- heldout = same-era fidelity to exact parent, not external task quality;
- no end-to-end production tok/s claim in this pack.
