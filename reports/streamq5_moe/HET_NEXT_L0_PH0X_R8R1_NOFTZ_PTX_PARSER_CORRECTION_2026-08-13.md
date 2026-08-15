# HET-NEXT L0 PH0X-R8-R1 — no-FTZ PTX parser correction

Date: 2026-08-13. CPU/text-only correction; no compile, allocation or device action.

The immutable R8 diagnostic JSON SHA is `c5df7a09ea13e4c29caa0d9acf40120131ae6a45033e73126f8563180f005ff2`; its immutable PTX SHA is `ec4789735f548123be0df3c2ff20c3e05c7b3741d9ed5f00b7b51eaeaa8ca7ae` (133,404 bytes). R8 compiled successfully with an empty log and found zero `.ftz`, 256 `mul.f32`, 256 `fma.rn.f32` and 34 `add.rn.f32` instructions. Its sole false predicate searched for a literal `, 8;`.

PTX encodes the source-level width-8 shuffle in the `c` operand as decimal `6175 = 0x181f`: clamp `0x1f` plus segment mask `0x18`. R1 parses exactly three `shfl.sync.down.b32` instructions with offsets 4, 2, 1; each has c=6175 and a register membermask. No source, compile output or threshold is changed. A corrected diagnostic pass may open a separately audited direct-PTX-load execution; it does not alter the formally negative R7 result.
