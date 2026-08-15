# PORT80B-T0Y-P canonical router performance report

Date: 2026-08-13

## Verdict

`performance_negative`, independently verified from 80 stored AB/BA validation pairs. The test phase remained closed.

Correctness replay against the independently verified T0Y-R raw tensors passed before timing. Resident validation results:

| Arm | p50 | p95 |
|---|---:|---:|
| Official CUDA router path | 0.130096 ms | 0.375082 ms |
| Canonical logits + deterministic IDs | 0.533552 ms | 0.603822 ms |
| Candidate/reference | 4.101218× | 1.609843× |

The frozen validation opening gates were p50 ratio at most 4.0 and p95 ratio at most 5.0. The p50 gate failed by 0.101218 ratio points, so the 240-pair test correctly did not run. No block-size, reduction-tree or gate retuning followed.

This measurement is favorable to the candidate because it does not compute selected probabilities, renormalized weights or BF16 weight output. Despite that advantage, the serial exact accumulation is too slow under the frozen p50 gate.

## Interpretation

T0Y-R remains a useful exact cross-backend oracle and proof of constructibility. T0Y-P shows that this naïve fixed-order implementation is not a practical replacement. A future optimized deterministic tree must preserve the exact arithmetic contract and be evaluated as a new preregistered hypothesis.

## Claim boundary

Resident 16-row component timing only. No layer/full-model throughput, quality, energy or deployment claim.
