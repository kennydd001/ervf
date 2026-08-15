# Pure GPTQ no-sync equivalence — attempt D lock

Locked after attempt C failed and before attempt-D outputs. The official per-matrix tensor shapes, 80-candidate order, candidate arithmetic, 128-value error reductions, strict-improvement comparisons, scale choices, Hessian, and sequential GPTQ updates remain unchanged. Only `if torch.any(mask): indexed assignment` is replaced by the exactly equivalent `torch.where(mask, new, old)`, removing host synchronization. Same 30 matrices; pass requires zero code and zero BF16-scale-bit mismatches.
