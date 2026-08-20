# Agent handoff — S100 Phase 25 H8 Best-of-All

Goal: execute the supplied Phase25 pack against the exact Nemotron 3.5 Lightning snapshot and publish only measured evidence.

Non-negotiable rules:

- Keep Phase24 frozen as the parent.
- Do not substitute another model/checkpoint or quantization.
- Do not reopen Phase24 attention/router/shared-M4 candidates.
- Do not promote any H8 arm that fails full state parity.
- Treat NVRTC/OOM/import failures as technical/feasibility results, not hypothesis failure.
- Never report 100 tok/s from route-count projections; only measured H8 wall time qualifies for target-only claims.
- Never report `S100_SINGLE_ACHIEVED=true` from this pack; Phase25 is target-verifier-only.

Primary outputs:

- `pro_research/results/s100_phase25/S100_PHASE25_PREFLIGHT.json`
- `.../S100_PHASE25_SCREEN_PARENT_CTX1024.json`
- `.../S100_PHASE25_SCREEN_<VARIANT>_CTX1024.json`
- `.../S100_PHASE25_STATE_CHECK_<VARIANT>.json`
- `.../S100_PHASE25_SELECTION.json`
- `.../S100_PHASE25_THERMAL_ADJUDICATION.json` when thermal gate opens
- `.../S100_PHASE25_PROFILE.json`
- `.../S100_PHASE25_SUMMARY.json`
- `reports/S100_PHASE25_RUN_REPORT.md`

Target branch: `agent/s100-phase25-h8-best-of-all`.
