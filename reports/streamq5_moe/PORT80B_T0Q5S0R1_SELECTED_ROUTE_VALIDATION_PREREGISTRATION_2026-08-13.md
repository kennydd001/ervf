# PORT80B-T0Q5-S0-R1 — standalone selected-route validation repair

S0-R1 preserves the S0 validation-only science, 252-expert union, D2-R3 inputs, codec and unchanged `0.08` routed/shared thresholds. It supersedes the immutable S0 implementation NO-GO and remains closed pending source audit.

The standalone runner and independent verifier shall both rerun the exact full-16 source-BF16 and Q5 graphs from official shard bytes and D2-R3 routes. Only routed, shared raw and gate-first `sigmoid(shared_gate_linear) * shared_raw` are scored; no complete/layer reconstruction.

Exactly 759 ordered matrix evidence rows are required: ascending selected routed IDs, then shared ID 512; projection 0 gate, 1 up, 2 down. Each row includes source/hash, codes/scales/decoded hashes, group count, zero-group count from `source.abs().amax(group)==0`, observed q min/max, field31 absence, decoded weight max-abs and rel-L2. Codes/scales are never persisted. Decoded matrices may be cached in RAM only and are released before final cleanup.

Controls use a real checker that parses requested/presented expert, projection, shape and expected codes/scales digests. Wrong expert is isolated to the fixed route occurrence. Projection swap is explicitly graph-wide for that expert. Shared-down code mutation is explicitly graph-wide; selection is independently row-major first source q nonzero with nonzero shared activation at the fixed row, mutation is exactly one q step toward zero, stored digest remains expected-original and must reject. Retain unsafe raw routed or shared raw and gated arrays; verifier reconstructs all.

Runtime is one-thread deterministic CPU, fixed affinity from the locked dependency contract, MKLDNN enabled, highest matmul precision, flush-denormal false, no autocast/inference mode true, CUDA uninitialized. Require start available RAM >=16 GiB, every-stage and final/cleanup available RAM >=2 GiB, Windows peak working set <=12 GiB, raw/result/commit/failure retained bytes <=512 MiB.

Successful output is a create-new raw/result/commit bundle: temp writes, file fsync, rename only if finals absent, directory fsync, marker last. Startup recovers/quarantines any uncommitted finals/temps. Failure is an atomic create-new temp/fsync/rename JSON after cleanup. Result includes exact raw manifest/schema/finiteness and cleanup/resource evidence.

Outcome remains `selected_route_validation_positive` (never pass), or a named negative/blocked/invalid. No preflight or physical execution before implementation audit.
