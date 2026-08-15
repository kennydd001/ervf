# T0R12D2R3 independent artifact audit and interpretation

## Verdict

The completed R3 artifact is a valid, reproducible diagnostic. All 15 independent audit checks pass: raw SHA-256, exact 1,752-tensor schema, per-tensor manifest, finiteness, stored stage/repeat/interpretation metrics, direct official router tuple, router arithmetic, tie evidence, provenance, input-file bindings, source identities, RAM bounds, and the diagnostic-only boundary.

This is not a scientific pass for Q5, a bank, performance, GPU execution, model quality, or 80B deployment. It is a four-prompt, layer-0, CPU-BF16 localization result.

## Independent stage recomputation

Each row below summarizes 64 comparisons: four prompts times prefix lengths 1 through 16. The compared value is the same final token from a length-16 whole-sequence call and a fresh length-*n* prefix call.

| Stage | Divergent comparisons | Different elements | Maximum BF16 ULP | Maximum absolute error | Divergent prompt/lengths |
|---|---:|---:|---:|---:|---|
| input norm | 0/64 | 0 | 0 | 0 | none |
| GDN output | 2/64 | 4 | 1 | 3.8146973e-06 | p0/n1; p1/n1 |
| post-attention norm | 0/64 | 0 | 0 | 0 | none |
| router logits | 0/64 | 0 | 0 | 0 | none |
| router weights | 0/64 | 0 | 0 | 0 | none |
| router IDs | 0/64 | 0 | n/a | n/a | none |
| routed experts | 10/64 | 33 | 16 | 2.44140625e-04 | p1/n3,5,9,11; p2/n2,8,9,11; p3/n8,9 |
| shared expert | 1/64 | 1 | 14 | 2.6077032e-08 | p1/n1 |
| shared gate | 0/64 | 0 | 0 | 0 | none |
| layer output | 5/64 | 16 | 16 | 2.44140625e-04 | p1/n3,5; p2/n2,8; p3/n8 |

Important causal boundary: the two tiny GDN differences do **not** survive the post-attention norm, which is bit-exact in all 64 comparisons. Consequently, they cannot explain the later expert/output mismatches. The observed layer-output mismatches coincide with routed-expert mismatches. Additional expert mismatches at p1/n9,n11, p2/n9,n11, and p3/n9 are rounded away by the final residual addition; this is consistent with the exact layer outputs there.

The single shared-expert difference at p1/n1 is also eliminated before the layer output and occurs while shared gate, router, and layer output remain exact.

## Determinism and cache

The two same-length repeats—prompt 1 at lengths 16 and 3—are bit-exact across all ten retained stages plus both cache states. This supplies direct evidence against run-to-run nondeterminism for those two sampled calls, but it is not an exhaustive determinism proof for every prompt and length.

All eight whole-versus-prefix16 cache comparisons are exact: four convolution states and four recurrent states. This proves that two fresh calls with the same complete length-16 input produce identical final cache state for these prompts. It does **not** test incremental cache reuse against a whole-sequence call at lengths 1–15, because the diagnostic deliberately creates a fresh cache for each prefix and retains only the final cache state.

## Router conclusion

All 64 whole-versus-prefix final-position comparisons are exact for router logits, normalized top-10 weights, and top-10 IDs. Across all 70 captures and 627 token rows, the independent audit also reproduces the direct second gate call, softmax/top-k arithmetic, BF16 weights, top-11 IDs/logits, margins, and tie masks.

Ties are common—112 token rows contain a multiway boundary tie and 80 have zero top-10/top-11 margin—yet no whole/prefix route mismatch occurs. Therefore, the original prefix mismatch is not a router-tie or router-recomputation failure on this frozen CPU execution.

## Safe scientific conclusion

The earlier `fresh-cache prefix mismatch` is real but localized: under the frozen single-thread CPU-BF16 backend, whole-sequence and fresh-prefix calls can produce sparse, deterministic, shape-dependent numerical differences inside the official routed-expert computation even when the post-GDN normalized inputs, router logits, weights, and expert IDs are bit-identical. Five of 64 sampled final-token comparisons retain a layer-output difference; the worst observed layer-output discrepancy is 16 BF16 ULP, 2.44140625e-04 absolute, and 1.2754377e-04 relative L2.

This refutes an exact-bit whole-versus-prefix oracle for this official CPU layer-0 path. It does not show a semantic model failure, incorrect cache semantics, incorrect routing, Q5 quality, or deployment viability. Any later Q5 gate must therefore compare both candidates against the same frozen official execution shape, or use a tolerance justified independently of these observed outcomes; it must not retroactively convert these diagnostic maxima into pass thresholds.

## Artifacts

- Raw safetensors: `reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors`, SHA-256 `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`, 171,696,126 bytes.
- Result JSON: `reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_result.json`, SHA-256 `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`.
- Independent audit JSON: `reports/streamq5_moe/PORT80B_T0R12D2R3_INDEPENDENT_ARTIFACT_AUDIT_2026-08-13.json`.
- Independent verifier: `scripts/streamq5_moe/audit_port80b_t0r12d2r3_artifacts.py`.

