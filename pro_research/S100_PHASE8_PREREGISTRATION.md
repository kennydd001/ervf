
# S100 phase 8 preregistration — static hot routed-down records

Date frozen: 2026-08-17

## Parent

`thr_0020` from phase 7 is frozen as the quality parent.

## Route-profile splits

- selection: phase-3 `_01` prompts, 64 teacher-forced targets per prompt;
- validation hit-rate report: phase-3 `_02` prompts, 128 targets per prompt.

The heldout `_03/_04` prompts are not used for cache selection. The cache is
arithmetic-exact, so no new model-quality selection occurs in this phase.

## Budgets

Total static `(layer, expert)` records:

- 64
- 128
- 192
- 256
- 320

Every record has the same physical size. Selection ranks all calibration
`(layer, expert)` counts globally by:

1. descending count;
2. ascending layer;
3. ascending expert ID.

No budget or ranking rule may be added after timing.

## Static-cache construction

For every selected pair:

- allocate one 2,806,272-byte device record;
- copy the full panel-major record from mapped host memory before capture;
- install a per-layer `expert_id -> static_slot` device map.

Preload time is setup time and excluded from decode timing. VRAM high-water is
included.

## Exact backend arms

For each budget and mode:

1. `BASE_A`: legacy `thr_0020`;
2. `CAND_A`: static cache;
3. `CAND_B`: static cache deterministic repeat;
4. `BASE_B`: legacy `thr_0020`.

Smoke additionally runs `BAD`, the same static cache with `bad_pick=1`.

## Gates

Correctness:

- baseline A/B token parity;
- candidate A/B token parity;
- candidate equals legacy;
- finite outputs;
- destructive control diverges;
- candidate runtime selection hash matches route-profile JSON.

Measurement:

- full ordinary arms each have at least 765 samples;
- baseline and candidate drift <=1.0 ms;
- VRAM <=7.8 GiB;
- candidate improves by at least 0.15 ms/token.

The fastest full arm passing every gate is selected. Otherwise legacy remains.

## Claim boundary

The cache is exact relative to phase-7 `thr_0020`; model quality is inherited
only after exact parity. S100-single still requires <=10.000 ms/token.
