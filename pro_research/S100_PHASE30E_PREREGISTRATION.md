# S100 Phase30E preregistration

## Frozen parent

- Code parent: `agent/s100-phase28-mirrorless-down` at `e759e22`.
- Active performance parent: promoted Phase24 H4.
- Context: 1024 for adoption; 128 and 4096 for generalization.
- Arithmetic, router, cache, LRU and state tolerances remain frozen.

## Candidate

Phase30E combines:

1. Phase27R `gather_y=4`, three gather/down batches and shared-stream overlap;
2. direct-L2 exact shared M4;
3. exact routed-UP dispatch in two launches, M1-2 and M3-4, indexed directly
   by device group rather than anchor scans over four mostly-empty grids.

## Required evidence

- exact IDs and deterministic candidate replay;
- SSM <= 5e-5 NRMSE, conv <= 1e-5, KV <= 5e-6, logits <= 5e-4;
- four fresh-process thermally rotated rounds;
- 16 measured H4 blocks and 8 warmups per arm per round;
- parent, shared-combined control and full candidate in every round;
- context 128 and 4096 exact generalization.

## Frozen adoption rule

Adopt only when state is green, all four round gains are positive, median
round gain is at least 5%, and bootstrap lower-95% gain is positive. Do not
move the gate after observing results.

## Reproduction

```powershell
.\RUN_S100_PHASE30E.ps1 -Action compile
.\RUN_S100_PHASE30E.ps1 -Action state
.\RUN_S100_PHASE30E.ps1 -Action thermal
.\RUN_S100_PHASE30E.ps1 -Action generalize
```
