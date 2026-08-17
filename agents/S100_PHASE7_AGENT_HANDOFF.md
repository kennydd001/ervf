
# S100 phase 7 agent handoff

Run only `RUN_ALL_S100_PHASE7.ps1` from the ZIP root.

The runner:

1. finds the completed phase-6 worktree;
2. applies an add-only patch;
3. tests packed-mirror exact parity and a destructive control in fresh
   processes;
4. selects packed only when full timing is exact, stable and at least
   0.15 ms faster;
5. evaluates every frozen phase-6 heldout candidate in its own process;
6. times every heldout-green candidate in fresh processes;
7. checks a packed candidate against the same candidate on legacy before
   accepting backend timing;
8. writes one summary;
9. updates `agent/s100-phase6-direct-down` and opens a draft PR when `gh` is
   authenticated.

Do not reinterpret phase-6 technical failures as quality failures. Do not add
new candidates to this heldout continuation. Do not call packed-mirror
microseconds a model speedup; only the complete fresh-process comparison counts.
