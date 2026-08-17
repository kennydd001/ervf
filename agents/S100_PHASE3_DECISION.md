# S100 phase 3 — decision after the integrated V18 run

Date: 2026-08-17

## Measured state

The full integrated V18 measurement promoted one profile:

- exact V18 midpoint: 19.5687 ms/token;
- FAST midpoint: 17.3798 ms/token;
- saving: 2.1889 ms/token;
- useful single-stream rate: 57.5381 tok/s;
- candidate VRAM: 6437 MiB;
- candidate repeat and all instrument/measurement gates: green.

FAST is not model-equivalent. Its greedy trajectories first diverged from V18
at token positions 43, 6 and 0 on the three timing prompts. It therefore opens a
quality/fidelity gate, not adoption.

The nested profiles reveal the load-bearing mechanism:

- MAMBA alone saved 0.93605 ms/token;
- SAFE (MAMBA + Q-FP8) saved only 0.66720 ms/token;
- FAST (MAMBA + Q-W4) saved 2.18890 ms/token.

Against separate matched baselines, Q-W4 adds roughly 1.25 ms beyond MAMBA,
while Q-FP8 anti-composes. This makes an unmeasured Q-W4-only profile mandatory:
it may retain most of FAST's speed with much less model perturbation.

FAST still needs 7.3798 ms/token to reach S100-single. Using the frozen traffic
model and the approximately 484.45 MB/token of resident dense bytes removed by
FAST, its first-order serial traffic floor is about 5.90 ms/token. S100 remains
physically possible, but roughly 64% of FAST's time above that revised floor
must still disappear.

## Phase-3 questions

1. How much does Q-W4 alone save in the real V18 graph?
2. Does FAST or Q-W4 preserve V18's distribution on a large, frozen,
   multi-domain reference trajectory?
3. Does reducing routed experts from K=6 to K=5 or K=4 buy real end-to-end time,
   and what V18-fidelity cost does it impose?
4. Do FAST and reduced routing compose, or does one mechanism erase the other's
   gain?

## Frozen phase-3 profiles

- `qfast`: only all six attention Q weights -> NVFP4 CEIL.
- `k5`: exact V18 weights, routed top-k 5.
- `k4`: exact V18 weights, routed top-k 4.
- `fast_k5`: FAST weights plus routed top-k 5.
- `fast_k4`: FAST weights plus routed top-k 4.
- `k1_control`: intentionally destructive quality-harness control.

## Important backend constraint

The V18 H-SCALE+B3 closure captures `top_k` when it is installed. A valid K5/K4
experiment must restore and reinstall that closure after changing top-k. Merely
changing `rt.top_k` and recapturing would mix K=4/5 loops with K=6 buffers/events
and is not a valid experiment. The phase-3 runner explicitly reinstalls the
closure for every top-k arm and again for BASE_B.

## Claim boundary

The fidelity suite compares candidates with exact V18 on frozen V18-generated
trajectories. It measures preservation of the current model, not ground-truth
perplexity or task accuracy. A passing profile still requires standard external
evaluations before it can be called quality-preserving.
