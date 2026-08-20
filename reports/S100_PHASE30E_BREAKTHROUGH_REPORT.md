# S100 Phase30E breakthrough report

## Outcome

Phase30E is locally adopted. It is the first current-parent H4 candidate in
this research chain that passes the frozen 5% threshold under the full
thermally rotated protocol.

The candidate combines three exact changes:

1. Phase27R's three-batch gather/down pipeline with shared-stream overlap;
2. direct-L2 shared M4, eliminating the 59 KiB activation staging allocation;
3. routed-UP device dispatch in two launches: M1-2 and M3-4, with direct group
   indexing instead of four grids whose blocks scan all 24 groups.

## Fidelity

- generated IDs: exact;
- deterministic candidate replay: exact;
- SSM NRMSE: 0.0;
- convolution NRMSE: 0.0;
- KV NRMSE: 0.0;
- logits NRMSE: 0.0;
- finite gates: green.

## Primary thermal adjudication — context 1024

| Round | Parent ms/H4 | Triple ms/H4 | Gain |
|---|---:|---:|---:|
| R1 | 75.2937 | 71.4482 | 5.107% |
| R2 | 77.1186 | 70.2168 | 8.950% |
| R3 | 75.1457 | 70.7688 | 5.825% |
| R4 | 75.5466 | 72.3715 | 4.203% |

- median round gain: 5.466%;
- bootstrap lower-95% gain: 4.203%;
- positive rounds: 4/4;
- median parent anchor: 75.4201 ms/H4;
- median triple anchor: 71.1085 ms/H4;
- absolute median-anchor saving: 4.3117 ms/H4;
- target-only rate at the median triple anchor: 56.25 tok/s.

The fixed adoption rule requires state green, four thermally balanced rounds,
all rounds positive, median gain at least 5%, and positive lower-95%. Every
condition passes. The adjudicator records `PHASE30E_ADOPTED=true`.

## Generalization

| Context | Parent ms/H4 | Triple ms/H4 | Gain | Triple tok/s |
|---:|---:|---:|---:|---:|
| 128 | 73.8755 | 68.0140 | 7.934% | 58.81 |
| 4096 | 83.8381 | 74.3072 | 11.368% | 53.83 |

Both contexts are token-exact and positive. The generalization adjudicator
records `PHASE30E_GENERALIZATION_GREEN=true`.

## What the experiments ruled out

- Four-token staged M4 with row16 tile1/2/4 is exact but 2–8% slower than the
  existing shared component.
- Shared M4 alone is materially faster as a component, but Phase27's overlap
  hides most of it end-to-end: only +0.80% incremental in the first smoke.
- One unified routed-UP kernel loses occupancy and is 20.6% slower.
- The winning routed-UP shape is the two-launch split; it saves 2.37 ms on the
  full routed-UP component with a +16.69% lower-95% gain.

## Repository status

The tested implementation was ported without arithmetic changes to normal
`pro_research` modules on the clean Phase28-derived branch
`codex/s100-phase30e-breakthrough`. `RUN_S100_PHASE30E.ps1` reproduces compile,
state, smoke, thermal and generalization protocols.

## Evidence files

- `pro_research/results/s100_phase30e/S100_PHASE30E_STATE_CHECK.json`
- `pro_research/results/s100_phase30e/S100_PHASE30E_ADJUDICATION.json`
- `pro_research/results/s100_phase30e/S100_PHASE30E_GENERALIZATION.json`
- `pro_research/results/s100_phase30e/S100_PHASE30D_GROUP_DISPATCH.json`
- `pro_research/results/s100_phase30e/S100_PHASE30B_OCCUPANCY_LADDER.json`
