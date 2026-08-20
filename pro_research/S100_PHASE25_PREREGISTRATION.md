# S100 Phase 25 — H8 Best-of-All Full Verifier preregistration

## Frozen parent

Phase 24 `phase24_best_of_all` is frozen. The official eight-token parent baseline is:

- Phase24 H4 + H4: **148.3404 ms / 8 tokens**.
- Absolute adoption gate: **<= 140.91 ms/H8**.
- 100 target-only tok/s gate: **<= 80.00 ms/H8**.

The exact Lightning snapshot remains `e8f3c7c4de75ad84fe1bcef95d38eca76214480b` and the 23 Phase24 H-SCALE resident MoE planes remain mandatory. Closed Phase24 attention/router/shared M4 arms stay off.

## H8 candidates

Three full-model H8 verifier candidates are tested in fresh processes:

1. `split4_route`: H8 route union across 48 routes, H8 resident down-gather, exact-style routed-up groups split into <=4-row chunks, Phase24-style route down reduction.
2. `direct8_route`: true M=1..8 H8 routed-up dispatch; M1..4 retains the H4 reduction geometry, M5..8 uses a register-bounded 32-lane/8-virtual geometry; Phase24-style route down reduction.
3. `direct8_groupdown`: direct M=1..8 routed-up plus an experimental group-down kernel that decodes each resident down weight once and updates all rows selecting that expert.

All candidates use one H8 graph, T8 Mamba recurrence/projections, exact causal attention offsets 0..7, production lm_head GEMV x8, and the same checkpoint weights.

## Correctness gates

A candidate is selection-eligible only if all eight output token IDs are exact and a full state comparison against two Phase24 H4 launches passes:

- deterministic candidate replay: exact token IDs;
- SSM NRMSE <= `5e-5`;
- conv NRMSE <= `1e-5`;
- KV NRMSE <= `5e-6`;
- eight-row logits NRMSE <= `5e-4`;
- finite outputs throughout.

No candidate with a red state gate can be adopted, regardless of latency.

## Screen and adoption gates

Screen ranking uses exact/state-green arms only. Thermal adoption opens only when the selected arm:

- is <= **140.91 ms/H8**, and
- is >= **5% faster** than the fresh same-run Phase24 H4+H4 parent.

Thermal adjudication is four alternating fresh-process rounds, 16 measured H8 windows each. Adoption requires:

- all measurements correct and positions aligned;
- median round gain >= 5%;
- median paired-block gain >= 5%;
- at least 3/4 rounds positive;
- parent and selected robust CV <= 5%;
- selected median-of-rounds <= 140.91 ms/H8.

## Claim boundary

`<=80 ms/H8` may set `S100_TARGET_ONLY_ACHIEVED=true`, but **must not** set `S100_SINGLE_ACHIEVED=true`. True single-stream S100 requires drafter, rejection, and fallback costs in the end-to-end path.
