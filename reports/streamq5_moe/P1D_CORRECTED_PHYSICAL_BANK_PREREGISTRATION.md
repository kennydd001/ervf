# STREAMQ5-MoE P1D - corrected physical full-bank preregistration

Locked on 2026-08-12 after the independently verified P0C quality and P1C
route-cache passes, and before producing any P1D physical-bank record. The
withdrawn P1B phase produced zero records and is not reused.

## Exact quantizer and decoder semantics

For every immutable BF16 expert weight, compute each group-128 maximum and
temporary scale `max(abs(group))/15` in FP32. Select signed codes `[-15,15]`
with round-to-nearest-even against that FP32 scale. Persist the scale as raw
BF16. The physical decoder reconstructs `code * float(BF16_scale)` and rounds
the output to BF16. This is exactly the P0C/P1C candidate.

## Record format

Layer files contain 128 experts in expert-major order; every expert contains
gate, up, and down records. Each matrix record has:

1. a fixed 64-byte little-endian `SQ5M` header (version, layer, expert,
   projection, bits, rows, columns, group size, payload sizes and CRC32);
2. row-major unsigned values `code+15`, packed little-order as eight 5-bit
   codes in five bytes;
3. row-major raw BF16 group scales;
4. zero padding to the next 4,096-byte record boundary.

Every matrix is exactly 1,011,712 bytes, every expert 3,035,136 bytes, every
layer 388,497,408 bytes, and the complete 48-layer bank 18,647,875,584 bytes
(17.3671875 GiB). Optional manifests do not count as bank bytes.

## Gates

- exactly 48 layer files, 6,144 experts, 18,432 records and 28,991,029,248
  codes;
- exact size and ordering; valid header, CRC, zero padding, code range, scale
  shape, and finite positive scales for every record;
- an independent decoder recomputes every Q5 code and raw BF16 scale bit from
  immutable source weights and matches every record;
- bank <=17.45 GiB, peak CUDA <=7.5 GiB, process RSS <=32 GiB;
- producer hash is locked before output and is append-only/resumable at whole
  layer boundaries.

P1D proves physical representation and exact source equivalence only. Measured
pinned transfer, Q5 kernel throughput, overlap, and integrated wall-clock are
separate phases.
