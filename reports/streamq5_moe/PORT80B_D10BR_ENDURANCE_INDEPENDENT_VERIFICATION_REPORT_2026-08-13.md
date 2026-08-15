# PORT80B-D10B-R independent endurance verification

Verdict: **PASS**. The independent CPU-only verifier passes **49/49 checks** and independently recomputes **19/19 frozen gates**. Failed checks: `[]`. Failed gates: `[]`.

## Exact recomputation

- Held-out stream: **10,000** cases, independently rebuilt SHA-256 `85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20`; max cold records per step **31** against **32** slots.
- Wall p50/p95/p99: **67.011950 / 69.047360 / 77.218634 ms**. CUDA-event p50/p95/p99: **66.975456 / 69.005653 / 77.097456 ms**.
- First/last-1,000 wall p95: **79.345895 / 69.031910 ms**; drift ratio **0.870012368**.
- State flags: **10,000/10,000 true**. Telemetry rows: **10,000**, in exact step order.
- Checkpoints: **101** on the frozen schedule; **909** array summaries. Digest uniqueness by array: `{'routed_capture': 101, 'routed_down': 101, 'shared_down': 1, 'attention': 65, 'delta': 60, 'kv_state': 42, 'recurrent_state': 64, 'conv_state': 33, 'composed_state': 101}`.
- Page telemetry: **703** samples; maximum Page Reads/sec **1020.738163**. Maximum Pages Input/sec was **8225.523605** and remains diagnostic, not the frozen gate.
- RAM: before **55,108,874,240 B**; after first touch **5,704,151,040 B**; minimum sampled during endurance **4,564,619,264 B**. First-minus-last sampled availability is **-1,024,204,800 B**.
- VRAM: post-allocation **2,863,661,056 B**; minimum sampled **2,844,786,688 B**.
- Dense checksum: expected/observed **6102583693077053440 / 6102583693077053440**; runtime sentinels `[165, 165]`.
- Host lifecycle: **48/48** registration and **48/48** unregister rows clean; **48,563,159,040 B** registered; no unregister failures.

## Provenance and retry/retune audit

The raw result locks the current preregistration, runner, CPU preflight, D10A2-R2 component result and the bank payload SHA declared by the manifest. The preflight locks the manifest file, all 48 route tensors and the frozen CPU unit/audit evidence. This verifier confirms the current bulk bank exists at the exact frozen **49,925,652,480 B** size, but deliberately does not rescan/hash all 49.9 GB. The original D10B attempt failed before any physical action; its four artifacts match the immutable hashes and no original D10B GPU result exists.

All checked execution/resource/route/schedule assignments are AST-identical between D10B and D10B-R. After normalizing only the revision result identifier and report heading, the full `endurance_phase` AST is identical. This is strong artifact-level evidence of **no retune** and no prior GPU retry. Absolute proof that no unrecorded external run occurred is outside filesystem provenance.

## Replayability boundary

The 101 x 9 checkpoint rows retain shape, dtype, finite flag, poison count and SHA-256, not the array bytes. This verifier confirms exact schedule/route metadata, schema, digest format and uniqueness, but **cannot recompute those 909 hashes from the underlying tensors**. Likewise, the 10,000 state records are Boolean guards rather than raw state tensors. Consequently this is a valid pass under the frozen summary contract, but it is not an independent numerical replay or a cross-run determinism proof.

## Claim boundary

Synthetic shape-informed 10,000-step endurance on held-out P4D-shaped proxy routes and uniform Q5 payloads only; not checkpoint, natural routing, quality, production throughput or breakthrough evidence.
