
# Agent handoff — S100 phase 4

## Apply

Run `APPLY_S100_PHASE4.ps1` from the existing reboot worktree.

## First commands

```powershell
.\pro_research\RUN_S100_PHASE4_FRESH_TIMING.ps1 `
    -Profile all -Mode smoke

.\pro_research\RUN_S100_PHASE4_FIDELITY.ps1 `
    -Profile all -Mode smoke
```

After technically valid smoke timing:

```powershell
.\pro_research\RUN_S100_PHASE4_FRESH_TIMING.ps1 `
    -Profile all -Mode full
```

Full fidelity is automatically restricted to smoke-green profiles:

```powershell
.\pro_research\RUN_S100_PHASE4_FIDELITY.ps1 `
    -Profile all -Mode full
```

## Interpretation

- Never use a phase-3 K-profile midpoint saving.
- Candidate absolute times from phase 3 are exploratory only.
- FAST smoke fidelity is red and must not be called quality-preserving.
- A verifier PASS establishes result-file consistency, not model quality.
- The fresh timing comparison is still a performance experiment only.
- 100 tok/s means <=10.000 ms/useful single-stream token, or must be explicitly
  labelled aggregate.

## Next implementation trigger

When fresh timing and smoke fidelity are both available:

1. Prefer the fastest fidelity-green primitive.
2. If QFAST or MAMBA is individually red, open layer-selective calibration.
3. If K5/K4 is red, do not loosen gates; open per-layer/adaptive K.
4. If no fidelity-green arm saves >=1 ms, prioritize E2 grouped MoE and the
   exact-reranked lm_head experiment rather than more global quantization.
