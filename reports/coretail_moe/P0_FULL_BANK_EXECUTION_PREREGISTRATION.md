# CORETAIL-MoE — P0 full-bank execution preregistration

Locked on 2026-08-11 before any new pure-GPTQ full-bank code was produced.

The original CORETAIL hypothesis, gates, and the physical format proven on locked16 remain unchanged. The new source must be the independently verified 6,144-expert **pure GPTQ** bank governed by `reports/qwen_gptq_bank/P0_GPTQ_SEMANTICS_ERRATUM.md`; the historically mislabeled locked16 helper output is retained only as prior format-mechanics evidence.

## Required source gate

`reports/qwen_gptq_bank/p0_full_bank_verification.json` must report `full_bank_pass`, 6,144 experts, 18,432 matrices, exact packed-code round trips, finite/nonzero BF16 scales, and exact source/calibration/artifact hashes. No RTN or historical locked16 substitution is allowed.

## Immutable physical format

Use exactly the P0A format:

- core: 64-byte record header, raw BF16 scales, fixed little-endian nonzero bitmap per row, uint32 row byte offsets, signbits packed only over nonzeros, CRC32, and 4,096-byte record alignment;
- tail: `q=-2` flags only among negative core entries, blocks of 64 rows, zlib-9 or raw when smaller, 32-byte block indices, per-block and record CRC32, and 4,096-byte record alignment.

All headers, indices, padding, checksums, and fallback bytes count. The decoder must reproduce every integer code and every BF16 scale bit.

## Locked gates

- actual complete core file ≤ 5.95 GiB;
- actual complete tail file ≤ 0.90 GiB;
- all 28,991,029,248 codes and all 226,492,416 BF16 scales exact;
- `core + 0.7176275253295898 GiB INT4 trunk + 0.375 GiB BF16 KV at 4K + 0.75 GiB reserve` ≤ 7.9599609375 GiB reported VRAM;
- no uncounted fixed-width fallback.

If any gate fails, close CORETAIL and do not open P1. If all pass, only P1 exact fused-kernel work opens; no quality or end-to-end speed claim is implied by P0.
