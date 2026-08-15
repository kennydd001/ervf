# PORT80B T0-R7 official-CPU route reproducibility preregistration

T0-R7 asks only whether the pinned official CPU backend reproduces router IDs and selected native-BF16 weights bitwise in two separately launched clean processes on four newly locked 16-token domain prompts. Capture 1 and capture 2 must use the same pinned shard, loader, environment, CPU affinity, dependency hashes and reference implementation. Neither capture may read the other.

Primary gates: exact prompt replay; exact official checkpoint/header/source identities; runtime/dependency locks; raw finite tensors and exact manifests; exact cache schemas; fresh-cache prefix ladder; BF16 manual decomposition gate; exact per-token router ID and native-selected-weight equality between clean captures. Rank-10 boundary ties are allowed, but every tie set, selected subset, native-BF16 rank-10/rank-11 logit bit pattern and probability margin must be retained and independently classified. Cross-backend route invariance remains the immutable R4 negative and is not tested or claimed.

Capture 1 may build the real differentiated layer-0 Q5 prerequisite bank only after all reference gates. Capture 2 must byte-reconstruct and compare it without overwrite. The independent compare phase alone adjudicates candidate pass. Create-new raw/result/failure evidence, 16-GiB start RAM, 12-GiB peak working-set, 2-GiB reserve, CPU-only and no retry/retune remain mandatory.

This preregistration does not open execution. No GPU, registration, physical Q5 execution or final T0-P claim is authorized.
