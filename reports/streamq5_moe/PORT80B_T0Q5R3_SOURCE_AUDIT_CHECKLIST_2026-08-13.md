# T0Q5-R3 source-audit checklist

No runner/model/forward/bank action is authorized. Return GO only for runner/verifier implementation.

1. Verify actual pre-MLP residual capture and exact native-BF16 residual-first `torch.add` identity/candidate construction; no subtractive reconstruction.
2. Verify strict bitwise source-graph control for routed/shared/complete/layer, full-16 fused gate_up and official increasing-ID dispatch; no ULP waiver.
3. Audit exact FP64 loop metric definitions, zero-norm cases, BF16 word/ULP map and 32-row aggregation.
4. Audit exact 64-byte header, CRC chaining, zero reserved/padding, biased q+15/field31, BF16 scales, byte arithmetic and decoded-weight digest.
5. Require independent verifier source reread and source hash/requantized codes+scales/decode digest/header/CRC/padding/offset verification for all 1,539 records without calling builder helpers.
6. Audit canonical manifest core/envelope serialization and non-self-referential hashes.
7. Audit exact prompt all-pairs disjointness, quality scope/gates and deterministic baseline/substitution controls.
