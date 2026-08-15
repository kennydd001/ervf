# PORT80B-D6/D7 independent CPU verification

Verdict: **d6_negative_and_d7_strong_component_pass_verified_with_exactness_scope_limit**. All replayable checks: **True**. No GPU code was executed.

## D6 — verified physical negative

The 24 stored validation samples recompute to p50 **76.420639 ms** and p95 **77.073959 ms**. The frozen p50 opening gate is missed by **11.420639 ms**, so the 120-sample test correctly did not run. The 1,474,560 stored outputs report zero bit differences and equal full-output digests, but the output arrays were not retained for an independent digest replay.

## D7 — verified strong component pass

Validation recomputes to p50/p95 **49.062880/49.924383 ms**. The 120 once-only test samples recompute to p50/p95 **48.734545/49.977299 ms**, effective rate **19.473033 GB/s**, and frozen-shell projection **78.054526 ms**. All primary and strong gates recompute true. Strong margins are **5.022701 ms** on expert-plane p95 and **11.945474 ms** on the projected total.

## Exactness evidence boundary

The stored resident/staged digests agree and the stored comparison reports 1,474,560/1,474,560 bitwise-equal outputs. This is stronger than a scalar mismatch count, but the output arrays were not saved, so the digests cannot be independently regenerated. More importantly, all synthetic expert payloads contain the same Q5 codes and scales and the compute kernels ignore headers. Therefore numerical equality cannot reveal a wrong layer/expert selection. The audit independently reconstructed the route indices and scanned all 973,209,600 selected source bytes for each correctness token, but D7 is not a differentiated-weight routing-correctness proof.

## Provenance and scope

The full 49,925,652,480-byte bank hash, both preregistrations, both evaluator hashes and the manifest hash match. Current P6/P7/D2/D5 dependency files match their prior locks/results, though D6/D7 did not pin those dependency hashes inside their own JSON. D7 uses a 973,209,600-byte HBM work buffer and a 307/512 (about 60%) registered prefix. It does not prove a full bank, real checkpoint, natural routing, physical dense shell, end-to-end throughput, quality or endurance.
