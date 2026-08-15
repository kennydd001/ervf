# Pure GPTQ no-`torch.any` equivalence — attempt F lock

Locked after the attempt-E diagnostic and before attempt-F outputs. The diagnostic proved the Hessian inverse bit-exact and localized all initial divergence to FP32 scale values (maximum `3.725290298461914e-09`): the replica divided by Python scalar `3`, whereas pinned upstream divides by its CUDA `maxq` tensor. Attempt F uses the exact CUDA tensor operand and separately materializes `xmin1`/`xmax1`, while retaining the attempt-E operation order and unconditional identical mask-index writes. Same 30 matrices; zero mismatches required.
