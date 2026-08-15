# STREAMQ5-MoE P1B - physical full-bank preregistration

Locked on 2026-08-12 after the independently verified P1A cache pass and
before producing any Q5 physical-bank record.

## Quantizer and record format

The producer must reproduce the P0 quality candidate exactly: symmetric
per-row group-128 RTN Q5, round-to-nearest-even, integer codes `[-15,15]`,
BF16 scales, no clipping or calibration.

Layer files contain 128 experts in expert-major order; every expert contains
gate, up, and down records. Each matrix record is:

1. a fixed 64-byte little-endian header (`SQ5M`, version, layer, expert,
   projection, bits, rows, columns, group size, code bytes, scale bytes,
   payload CRC32);
2. row-major codes mapped to unsigned `code+15` and packed as a little-order
   5-bit stream, eight codes into five bytes;
3. row-major raw BF16 group scales;
4. zero padding to a 4,096-byte boundary.

Each record must be exactly 1,011,712 bytes; each expert 3,035,136 bytes; each
layer 388,497,408 bytes; and the complete 48-layer bank 18,647,875,584 bytes
(17.3671875 GiB). Optional manifests are outside bank bytes.

## Gates

- exactly 48 layer files, 6,144 experts, 18,432 records and 28,991,029,248
  codes;
- exact expected sizes and record ordering;
- every header, CRC, padding byte, code range, scale shape and finite/nonzero
  scale is valid;
- an independent decoder recomputes all Q5 codes and raw BF16 scale bits from
  the immutable BF16 source weights and matches every physical record;
- bank <=17.45 GiB, peak CUDA <=7.5 GiB and process RSS <=32 GiB;
- producer is append-only and resumable at complete layer boundaries.

P1B proves physical representation only. Kernel throughput, pinned transfer,
cache overlap and end-to-end tokens per second remain separate phases.
