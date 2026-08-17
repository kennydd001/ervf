# S100 phase 3 timing preregistration

Date frozen: 2026-08-17
Canonical tracked base: `pro-research@c839060`
Required additive parents: S100 reboot pack and phase-2 pack.

## Arms

Each profile runs in a fresh process with:

1. `BASE_A` — exact V18;
2. `CAND_A` — candidate, recaptured;
3. `CAND_B` — deterministic candidate repeat;
4. `BASE_B` — exact checkpoint/V18 restored and recaptured.

Profiles:

- `qfast`: Q-W4 CEIL only;
- `k5`: exact weights, routed top-k 5;
- `k4`: exact weights, routed top-k 4;
- `fast_k5`: FAST plus top-k 5;
- `fast_k4`: FAST plus top-k 4.

No component timing is converted into tok/s. Only the full graph result counts.

## Correctness and instrument gates

- BASE_A token ids equal BASE_B;
- CAND_A token ids equal CAND_B;
- candidate logits finite;
- profile manifest and expected top-k match;
- Q mixed dispatch executes when required;
- a top-k profile reinstalls H-SCALE+B3 with the candidate top-k;
- BASE_B reinstalls H-SCALE+B3 at top-k 6;
- invalid profile names are rejected by argparse.

Candidate divergence from BASE_A is reported and expected for approximate arms.

## Measurement gates

- baseline p50 drift <=1.0 ms;
- candidate p50 drift <=1.0 ms;
- candidate VRAM <=7987 MiB;
- full mode has at least 765 timed samples.

Profile-specific information gates:

- qfast: gain >=0.75 ms/token;
- k5: gain >=0.75 ms/token;
- k4: gain >=1.50 ms/token;
- fast_k5: candidate midpoint <=16.50 ms/token;
- fast_k4: candidate midpoint <=15.50 ms/token.

Missing these gates does not retroactively alter the measurement. It closes only
that exact profile as a phase-3 primary lever.
