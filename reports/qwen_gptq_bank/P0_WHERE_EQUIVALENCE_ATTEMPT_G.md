# Pure GPTQ `where` equivalence — attempt G lock

Locked while the already-proven attempt-F layer-0 checkpoint was running and before attempt-G outputs. Attempt G is identical to F, including CUDA-tensor `maxq` operands and separately materialized `xmin1`/`xmax1`, except mask-index writes are expressed as elementwise `torch.where`. This tests whether the earlier D failure was entirely the diagnosed scalar-operand issue. The same 30 matrices and zero-mismatch gate apply. A pass may be used only for subsequent layers; layer 0 remains the exact F output.
