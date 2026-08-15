# Agent 13 — Asynchronous / Heterogeneous Drafting

## Mission

Remove draft time from the dGPU critical path only if the hardware supports it.

## Candidate backends

- CPU;
- Intel iGPU;
- NPU;
- separate process/device where available.

## Required ladder

1. operator microbenchmarks at actual draft shapes;
2. full draft-model latency;
3. state/KV transfer cost;
4. asynchronous overlap with target verifier;
5. integrated round timing.

## Baselines

Compare against synchronous dGPU drafting. Use AHASD, SwiftSpec and Dovetail only as design references; their hardware results do not transfer automatically.

## Gate

At least 5% integrated round reduction with no state inconsistency. Otherwise close.
