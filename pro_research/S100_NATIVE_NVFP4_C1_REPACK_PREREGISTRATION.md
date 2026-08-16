# S100 native NVFP4 C1 — Blackwell scale-layout repack proof

Date: 2026-08-16
Branch: `pro-s100-nativefp4`
Parent evidence: C0B `92dc8eb9f001de521a77a02ada7b82b2304f7662`
Status: frozen before C1 execution.

## Question

C0B proved that all 5,935 audited Lightning NVFP4 weight/scale pairs are exactly logical group-16 once packed 2-codes/byte storage is accounted for. The remaining format delta is layout: checkpoint block scales are natural row-major `[M, K/16]`, while Blackwell block-scaled NVFP4 MMA consumes the documented `SWIZZLE_32_4_4` / 128x4 tiled scale layout.

C1 asks only:

> Can every Lightning NVFP4 scale tensor be mapped to that Blackwell scale layout and back **losslessly**, with only documented padding and no scale/code/value change?

No Tensor Core matmul, activation quantization, model quality or speed claim is allowed in C1.

## Primary-source layout contract

NVIDIA CUTLASS Blackwell documentation specifies:

- NVFP4 scale vector size = 16;
- one hardware scale basic block contains 128 rows x 4 scale factors = 512 bytes;
- within a 128x4 block, rows are interleaved in 32-row groups: for each `r in 0..31`, the byte groups for rows `r`, `r+32`, `r+64`, `r+96`, each across the four SF-K entries, are consecutive;
- multiple 128x4 blocks are arranged K-major;
- physical scale element count is `round_up(M,128) * round_up(ceil(K/16),4)`.

References:
- `NVIDIA/cutlass/media/docs/cpp/blackwell_functionality.md` — Scale Factor Layouts.
- CUTLASS `Sm1xxBlockScaledConfig<SFVecSize>` / `tile_atom_to_shape_SFA/SFB`.
- PyTorch `ScaleSwizzleMode::Swizzle32x4x4`, which documents the same 128x4 / groups-of-32 layout and padded element-count rule.

Frozen offset interpretation for a natural scale coordinate `(m, sf)` where `sf=k//16`:

```text
mb  = m // 128
r   = m % 128
r32 = r % 32
g32 = r // 32
kb  = sf // 4
sf4 = sf % 4

block = kb * ceil(M/128) + mb       # K-major basic blocks
inner = ((r32 * 4 + g32) * 4 + sf4)
offset = block * 512 + inner
```

Padding slots for `m>=M` or `sf>=K/16` are zero-filled and are not semantic scale values.

This mapping is frozen before payload inspection. If an actual NVIDIA API/layout cross-check later disproves it, C1 is `mapping_contract_failed`; do not silently change the formula after seeing payload results.

## C1 gates

### Structural, all 5,935 pairs

- `C1_G1_all_group16_parent`: C0B status is `format_counts_group16_packed_exact`, breakers=0.
- `C1_G2_all_shapes_padded_count`: for every pair, swizzled storage count equals `ceil(M/128)*128 * ceil(SFK/4)*4` and is never smaller than natural count.
- `C1_G3_mapping_in_bounds`: every tested natural coordinate maps inside that padded count.
- `C1_G4_mapping_unique`: no two natural coordinates collide. For very large shapes this may be proven by exact inverse-coordinate algebra plus deterministic boundary/stratified sampling rather than materializing a >100-MiB index set.
- `C1_G5_inverse_formula`: `inverse(offset(m,sf)) == (m,sf)` on all enumerated small/medium shapes and deterministic boundary+random samples on large shapes.

### Real payload round trip

Choose representatives before reading their payload bytes: at least one tensor from every distinct `(M,SFK)` scale shape, plus early/middle/late routed-expert examples when shape duplicates exist. For each selected scale tensor:

1. read the exact FP8-E4M3 scale bytes;
2. natural -> swizzled padded buffer;
3. swizzled -> natural;
4. require byte-for-byte identity.

Gates:

- `C1_G6_payload_scale_roundtrip_exact`: zero byte mismatches for every selected tensor.
- `C1_G7_padding_does_not_alias`: every semantic scale maps to a non-padding slot and padding extraction never contributes to inverse output.
- `C1_G8_global_scale_unchanged`: representative `weight_scale_2` F32 bytes are read unchanged; repack never touches them.
- `C1_G9_codes_unchanged`: representative packed E2M1 weight bytes are not rewritten by scale repack; their SHA-256 before/after reference is identical.
- `C1_G10_sampled_dequant_reconstruction`: deterministic sampled logical weights dequantize to identical IEEE bits before and after scale swizzle+inverse using the locked runtime rule `e2m1(code) * e4m3(scale) * f32(weight_scale_2)`.

## Overhead accounting

Report natural scale bytes, padded native scale bytes and padding overhead for every distinct shape and weighted over all 5,935 pairs. Padding is format overhead, not a quality cost.

Strong format gate:

- total native scale-layout padding overhead <= 5% of natural scale bytes.

If >5%, native FP4 remains technically possible but C2 must include the extra bytes in all roofline/performance accounting.

## Result classes

- `repack_lossless`: all correctness gates pass. Authorizes C2 native-matmul feasibility; no speed claim.
- `repack_lossless_high_padding`: correctness passes but >5% scale padding overhead.
- `mapping_contract_failed`: frozen SWIZZLE_32_4_4 mapping disagrees with an authoritative local API/layout cross-check.
- `repack_not_lossless`: any semantic scale/global/code/dequant reconstruction mismatch.
- `technical_failure`: environment/file failure before a valid decision.

## Next step if green

C2 is a separate preregistration. It must determine what native SM120 primitive is actually callable on this machine and, critically, whether its activation operand must also be FP4. If activation quantization changes target arithmetic, native FP4 cannot be inserted into the exact V18 verifier track without a new numerical/quality contract; it may still be valuable as an approximate MTP drafter.
