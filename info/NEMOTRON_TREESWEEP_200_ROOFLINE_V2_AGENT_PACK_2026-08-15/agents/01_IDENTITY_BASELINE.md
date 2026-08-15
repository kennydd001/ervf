# Agent 01 — Identity and Baseline Lock

## Mission

Establish exactly what “Nemotron 3.5 Lightning” is and reproduce the provisional AR/MTP measurements before opening any scientific branch.

## Required work

1. Hash every model/runtime artifact.
2. Record model ID, revision, tokenizer, chat template, layer pattern, routed/shared expert count, top-k, hidden size, Mamba/GQA configuration and quantization.
3. Inspect the MTP path: weights, number of positions, dependence structure, proposal algorithm and acceptance rule.
4. Rebuild with pinned compiler flags and record runtime commit.
5. Reproduce short, 128K and 262K AR latency where feasible.
6. Reproduce mean accepted draft tokens and per-depth acceptance distribution.
7. Decompose one target round into draft, Mamba, GQA, MoE, head, state commit and scheduler time.

## Controls

- Compare public Nano, NIM 3.5, Elastic and local Lightning metadata; do not infer identity from name similarity.
- Run deterministic prompt fixtures.
- Record exact context, batch, temperature and reasoning settings.

## Deliverables

- `P0_IDENTITY_BASELINE_PREREGISTRATION.md`
- `P0_IDENTITY_MANIFEST.json`
- `P0_BASELINE_MEASUREMENTS.json`
- `P0_FINAL_REPORT.md`

## Gates

Use the P0 gates in `EXPERIMENT_REGISTRY.yaml`. No later branch opens on a partial identity.
