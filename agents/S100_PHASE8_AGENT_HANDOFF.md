
# S100 phase 8 agent handoff

Run only the ZIP-root `RUN_ALL_S100_PHASE8.ps1`.

The runner must:

1. find the completed phase-7 S100 worktree;
2. apply the add-only patch;
3. profile route IDs on calibration and validation;
4. freeze static-cache selections for 64/128/192/256/320 records;
5. test every budget in independent smoke processes;
6. run full timing only after exact smoke parity;
7. retain a budget only if full A/C/C/B is exact, stable and at least
   0.15 ms faster;
8. inherit quality only from the frozen phase-7 `thr_0020` parent;
9. write `S100_PHASE8_SUMMARY.txt/json`;
10. update the existing research branch and draft PR.

A static hit-rate estimate is not a latency claim. A microkernel result is not a
model result. S100 means complete measured latency <=10.000 ms with quality
green.
