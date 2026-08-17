# Agent handoff — S100 phase 3

## Current proof

- Exact V18: 19.6046 ms/token, 51.008 tok/s.
- Integrated FAST: 17.3798 ms/token, 57.538 tok/s, 2.1889 ms saved.
- FAST is deterministic but not token-identical; it is only a performance
  candidate until the fidelity gate runs.
- Large BF16 attention Q/O matrices showed approximately 15.4–15.8x useful-row
  throughput at M=16, so true aggregate batching remains open.

## Run order

From the existing S100 reboot worktree after applying `S100_PHASE3.patch`:

```powershell
.\pro_research\RUN_S100_PHASE3_TIMING.ps1 -Profile all -Mode smoke
.\pro_research\RUN_S100_PHASE3_TIMING.ps1 -Profile all -Mode full
```

Review the full results before spending time on every quality profile. The
minimum decisive fidelity order is:

```powershell
.\pro_research\RUN_S100_PHASE3_FIDELITY.ps1 -Profile qfast -Mode smoke
.\pro_research\RUN_S100_PHASE3_FIDELITY.ps1 -Profile fast  -Mode smoke
.\pro_research\RUN_S100_PHASE3_FIDELITY.ps1 -Profile qfast -Mode full
.\pro_research\RUN_S100_PHASE3_FIDELITY.ps1 -Profile fast  -Mode full
```

Then run K5/K4 fidelity only for timing-positive arms. Combination profiles are
last, because their individual causes must be understood first.

## Interpretation

- QFAST timing-positive and fidelity-green: prefer it over FAST unless FAST's
  extra Mamba conversion also passes and materially improves speed.
- FAST fidelity-red: do not call 57.5 tok/s quality-preserving. Rank Mamba
  layers by sensitivity or retain QFAST only.
- K5 fidelity-green: integrate it with the best weight profile.
- K4 fidelity-green: it is likely the largest remaining single-stream MoE lever.
- K5/K4 fidelity-red: proceed to activation-neuron sparsity and a compiled model
  recovery phase rather than quietly relaxing gates.
- FAST_K4 below 15.5 ms still leaves >5.5 ms to S100. The next route is
  downflow/cache plus exact-reranked lm_head, then hybrid sublayer pruning or a
  block verifier.

## Backend warning

Do not assume `scaled_grouped_mm` availability means SM120 grouped NVFP4 MoE is
correct or fast. The current ecosystem has reported SM120 grouped-FP4
correctness/autotuning failures. Any E2 grouped path needs:

1. synthetic nonuniform known values;
2. real checkpoint rows;
3. an independent reference;
4. a sabotage/control arm;
5. cold timing;
6. only then integration.

## Required outputs

Timing:

```text
pro_research/results/S100_PHASE3_TIMING_QFAST.json
pro_research/results/S100_PHASE3_TIMING_K5.json
pro_research/results/S100_PHASE3_TIMING_K4.json
pro_research/results/S100_PHASE3_TIMING_FAST_K5.json
pro_research/results/S100_PHASE3_TIMING_FAST_K4.json
```

Fidelity results additionally produce an NPZ metrics file and a markdown
side-by-side rollout report. Return the JSON, verifier output and markdown
review, not only the one-line status.
