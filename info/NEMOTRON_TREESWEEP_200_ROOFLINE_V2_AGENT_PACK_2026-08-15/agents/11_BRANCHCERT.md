# Agent 11 — BranchCert / Proof-Carrying Off-Path Precision

## Mission

Reduce verification cost of off-path tree nodes without changing target semantics.

## Idea

Represent NVFP4 weights as an exact core plus residual pages. Evaluate low-cost bounds first. Fetch exact pages only when intervals cannot certify:

- router top-k;
- ReLU² sign/rounded activation;
- rounded linear output;
- candidate acceptance/rejection.

## Rules

- use outward-rounded interval arithmetic;
- zero false certificates;
- no learned risk gate as a substitute;
- start with LM-head/acceptance and fc2, not the hardest pre-nonlinearity path;
- measure metadata and bound-compute cost.

## Gate

At least 30% of off-path tail bytes must be skipped and integrated round time must improve at least 5%. Otherwise close the branch.
