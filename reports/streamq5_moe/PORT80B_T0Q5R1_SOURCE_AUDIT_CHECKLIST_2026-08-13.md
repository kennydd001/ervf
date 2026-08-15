# T0Q5-R1 independent source-audit checklist

Execution remains closed. The audit must inspect the preregistration, prompt generator and codec contract and issue GO/NO-GO before prompt-lock generation, official model load, forward, shard payload scan or bank creation.

Required checks:

1. Confirm official revision/shard identity and that only layer-0 routed/shared expert tensors are in scope.
2. Recompute all wire byte arithmetic: group 128, 5 bits, BF16 scales, 64-byte header, 4,032-byte padding, 675,840-byte matrices, 2,027,520-byte expert triples, 513 records and 1,040,117,760-byte bank.
3. Inspect biased `field=q+15`, round-to-nearest-ties-to-even, zero scale `1.0`, field-31 rejection, little 8-in-5 packing and exact little-endian BF16-scale byte order.
4. Confirm fresh prompt candidates are semantically natural and disjoint from T0-R12 without output/route-dependent filtering; require canonical prompt-lock bytes before outputs.
5. Confirm whole-sequence official capture avoids the known prefix/full shape comparison as a correctness gate.
6. Assess whether the predeclared quality thresholds are meaningful and sufficiently strict before any Q5 output is observed.
7. Require executable runner/verifier to bind prereg, generator, prompt lock, codec contract, official source files, environment, shard, previous D2-R3 result/verifier, and its own sources by SHA-256.
8. Require independent bank parse/source reconstruction, independent metric recomputation, all control adjudication, create-new failure evidence and the stated resource/disk bounds.

The codec source is an auditable contract, not an experiment runner. Execution remains closed until an independently reviewed runner and verifier bind and implement this contract.
