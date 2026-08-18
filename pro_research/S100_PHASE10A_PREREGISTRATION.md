# Phase 10A preregistration
- Parent: Phase-9 current-map QFAST + alpha=0.0003.
- Calibration: `_01`, 64 teacher-forced targets.
- Validation: `_02`, 128 targets; report-only.
- Budgets: 8/16/24/32/40/48 MiB panel code.
- Exact candidate: routing, masks, H-SCALE byte, FMA order, chunk reduction and
  route accumulation unchanged. Only cached code-byte source changes.
- Full promotion gate: >=0.15 ms/token.
- No component result counts as S100.
