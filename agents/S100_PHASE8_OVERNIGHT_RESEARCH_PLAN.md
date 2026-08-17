# S100 phase 8 — overnight Arc 140T adjudication

Date frozen: 2026-08-17.

The overnight campaign decides whether Arc 140T should enter the latency-critical QFAST routed-down path. It compares current RTX H-SCALE downflow with Arc over the same real QFAST geometry and expert records.

## Real-data coverage

Four deterministic causal snapshots are captured after decode offsets 0, 1, 4, and 16. Each snapshot contains all 23 MoE layers and their six actual selected experts, route weights, ReLU2 activations, masks and panel-major NVFP4 records.

## Direct comparator

The current QFAST RTX down path is measured on the same snapshot offsets using the live phase-5 closure.

## Full-bank anti-cache control

One complete real down-expert bank is kept in one persistent Intel-GPU buffer. Random six-expert route sets span the bank with a 128 MiB cache scrub before timed kernels. The cold/warm factor is applied conservatively during adjudication.

## Frozen decision

ADE_GO requires strict N=6 correctness, pressure-adjusted Arc+bridge >=10% faster than measured RTX down-only with no negative snapshot, median QFAST contention <=5%, first-to-last Arc drift <=10%, and >=3 independent reruns. ADE_NO_GO if correctness fails, Arc is >=10% slower, or sustained QFAST regression exceeds 15%. Everything else is ADE_BORDERLINE.

No component projection counts as S100. GO only opens end-to-end integration.
