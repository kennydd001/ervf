# S100 phase 6 agent handoff

Run only the ZIP-root `RUN_ALL_S100_PHASE6.ps1`.

The runner must:

1. locate the completed phase-5 reboot worktree;
2. apply the add-only patch;
3. prove exact backend parity and destructive-control power;
4. measure all exact backends in fresh processes;
5. run the fixed validation grid;
6. read heldout only for validation-green candidates;
7. time only heldout-green candidates;
8. write `S100_PHASE6_SUMMARY.txt/json`.

Do not call a direct-host microbenchmark an end-to-end speedup. Do not loosen
heldout gates. The phase-5 strict gate remains historical evidence; phase 6 is
a new confirmatory experiment whose candidate grid is frozen before heldout.
