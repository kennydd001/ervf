# S100 Lightning Phase 16 preregistration

Target identity must match the fresh Phase-15 Lightning signature.

## Stream adjudication

For TC2 on K/V/O:
- asynchronous producer sentinel;
- real teacher-forced shadow calls before authoritative ERVF;
- legacy, context-first, sync-control paths;
- record NRMSE, cosine, max absolute error, stream pointers and repeats.

A stream bug is confirmed only if context-first or sync-control improves
legacy error by >=10x and reaches max NRMSE <=1e-4.

## Layer screen

Each of the 18 K/V/O matrices is substituted alone under TC1 and TC2 on a
frozen 4-prompt x 16-token screen. A screen pass requires:
- top1 >=0.95;
- top5 >=0.99;
- mean CE delta <=0.05;
- mean coarse KL <=0.03;
- finite.

Greedy cumulative safe sets are built on screen data only.

## Full quality

Selected sets run full calibration. Strict calibration pass is required before
validation; strict validation before heldout. Heldout uses the Phase-15
official gates and deterministic repeat.

## Lightning verifier and DFlash2

The ordinary perfect-draft B={2,4,8} verifier, route union, hidden-state proxy,
selector proxy and resident-memory screen are rerun on Lightning. No Nano
result contributes to a gate.

DFlash2 training opens only if:
- a measured Lightning verifier leaves positive S100 draft budget;
- a resident draft configuration fits with reserve;
- Lightning suffix/lattice transfer evidence is green.

Technical failures remain null.
