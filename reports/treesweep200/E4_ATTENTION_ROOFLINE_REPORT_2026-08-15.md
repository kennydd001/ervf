# E4 — Attention Roofline Recovery — rapport (2026-08-15)

Preregistratie: `E4_ATTENTION_ROOFLINE_PREREGISTRATION_2026-08-15.md` (bevroren
vóór metingen; geen poort verruimd). Runner:
`scripts/treesweep200/e4_attention_roofline.py`. Resultaten:
`E4_ATTENTION_ROOFLINE_RESULTS.json`. Protected manifest na E4:
0 modified / 0 removed (`protected_verification_after_e4.json`).

## Claim boundary

Alle getallen hieronder zijn componentmetingen op de geïsoleerde
attention-kernel met random fp8-KV-data, tenzij expliciet "in-lus" staat. Niets
hieronder is een tok/s-claim. De in-lus adoptiemeting (G-E4-T1) is NIET
uitgevoerd — zie handoff.

## Poorten-uitslag

| Poort | Eis | Uitslag | Status |
|---|---|---|---|
| G-E4-F1 | v1-fit R² ≥ 0,99 | R² = 0,9981 (slope 19,89 ms/GB, intercept 0,010 ms) | PASS |
| G-E4-P1 | 6 profielen × 5 contexten, monotonie, ≥90% verklaard | compleet; addr_scan 0,482 / qk 2,179 / softmax 2,007 / full — zie JSON | PASS |
| G-E4-C1 | rel_l2 ≤ 3e-4, determinisme | v2: 1,26e-5; v3/v4: **bitwise identiek** aan v1; v6/v7: 4,8e-6; determinisme True | PASS |
| G-E4-S1 | ≥100 GB/s @262144 (≤1,342 ms/laag) | beste: v4 = 2,304 ms = 58,3 GB/s | **FAIL** |
| G-E4-S2 | ≥169 GB/s (≤0,794 ms/laag) | idem | **FAIL** |
| G-E4-T1 | in-lus ≤6 ms @262100 + token-pariteit | niet uitgevoerd (adoptie v4 voorbereid, meting open) | OPEN |

## Gemeten (full path = kernel + combine, ms/laag)

| t | bytes/laag | v1 | v2 | v3 | v4 | v6 | v7 | raw_scan |
|---|---|---|---|---|---|---|---|---|
| 64 | 32,8 KB | 0,035 | 0,039 | 0,031 | 0,030 | 0,031 | 0,029 | — |
| 4096 | 2,1 MB | 0,126 | 0,168 | 0,116 | 0,084 | 0,093 | 0,082 | 0,029 |
| 32768 | 16,8 MB | 0,312 | 0,379 | 0,296 | 0,287 | 0,316 | 0,282 | 0,198 |
| 131072 | 67,1 MB | 1,365 | 1,939 | 1,235 | 1,218 | 1,513 | 1,593 | 0,690 |
| 262144 | 134,2 MB | 2,803 | 3,715 | 2,509 | **2,304** | 2,605 | 2,401 | 0,934 |

Effectief @262144: v1 47,9 → v4 **58,3 GB/s** (−17,8% tijd). Adres-patroon-
plafond (addr_scan, zelfde indexing zonder rekenwerk): 0,482 ms = 278 GB/s.

## Kandidaten (allemaal additief in `src/moe_lab/lightningstream_nemotron/gpu_kernels.py`)

- **v1** `attn_decode_warp_fp8_gqa` — huidige productiekernel (lane=4 dims,
  16-head butterfly, shared-LUT decode).
- **v2** `..._gqa2` — 2 lanes/head. Was nooit geregistreerd; bevatte een
  out-of-bounds writeback-bug (`base = hf*64+u*4` → `acc[64..127]`; gerepareerd
  in deze fase). Correct na fix (1,3e-5) maar structureel trager: 16×
  redundante LUT-decode → shared-pipe-bound.
- **v3** `..._gqa3` — v1 + hardware fp8→f16x2 `cvt` (sm_89+) i.p.v. LUT +
  double-buffered loads. **Bitwise identiek aan v1** (e4m3 is exact in f16;
  zelfde operatievolgorde). −10,5% @262K.
- **v4** `..._gqa4` — v3 + 2 posities/warp-iteratie (ILP). **Bitwise identiek
  aan v1**. Beste kandidaat: −17,8% @262K. Geregistreerd + wrapper
  `attention_fp8_gqa4`.
- **v6** `..._gqa6` — packed-fp32 (`fma.rn.f32x2`) + q in registers. Correct
  (4,8e-6) maar trager dan v4: registerdruk (qr[16][2]=64 regs) + pairing-movs.
- **v7** `..._gqa7` — v4 + f32x2 + shared-float2-q. Correct, ~v4-niveau.

## Nieuwe hardware-kennis (sm_120, RTX PRO 2000 Blackwell, 26 SM)

- `fma.rn.f32x2` bestaat op sm_120 en is **bitwise gelijk aan scalair fmaf**
  (2048/2048 cases, incl. extreme exponenten; een eerdere "mismatch" was een
  probe-bug: float64-array doorgegeven aan een float*-kernel). Gemeten
  throughput: 1,63× scalair FP32, niet 2×.
- `redux.sync.add.f32` is **NIET** ondersteund op sm_120 (ptxas: alleen
  sm_100a-familie). Butterfly-reducties kunnen dus niet door één instructie
  vervangen worden.
- De kernel is **issue-bound** (~1,2 warp-inst/cyclus/SM), niet HBM-bound:
  ablaties @262144 (v4-structuur): full 2,27 / zonder shuffles 1,42 /
  zonder exp 2,19 / zonder PV 1,75 ms. Shuffles ≈ 0,85 ms (80 warp-shuffles
  per positie-beoek × 524288 bezoeken ≈ 42M, shuffle-pipe ~1/cyclus/SM).

## Waarom S1/S2 niet haalbaar bleken in exact-fp32 (analyse, geen poort)

Zuiver FP-werk per positie-bezoek = 16 heads × 128 dims × 2 (dot+PV) × 2 flop
= 8192 flop fp32. Bij gemeten f32x2-piek 12,8 TF/s → 0,34 ms absolute vloer;
met shuffles/softmax/decode/loads (≥50% van de instructiestroom) is een
realistische exacte vloer ~1,2–1,5 ms ≈ 90–110 GB/s. S1 (100 GB/s) zit aan de
rand van het denkbare; S2 (169 GB/s) ligt er structureel boven. Verder gaan
vereist fp16/tensor-core-paden (breken exactheid: q in fp16 geeft ~1,5%
logit-fout — verworpen) of de transpose-reduce (31 i.p.v. 80 shuffles; geschat
~0,5 ms winst → ~1,8 ms ≈ 75 GB/s — onvoldoende voor S1, niet gebouwd).

## Besluit

- **Adoptie-kandidaat: v4** — bitwise identiek aan v1, −17,8% @262K, −33%
  @4K. Voorspeld in-lus effect (component→schatting, GEEN tok/s-claim):
  attention 6×(2,80+0,10) → 6×(2,30+0,10) ms bespaart ~3,0 ms/token @262100.
- E4-status in registry: `gate_failed` (S1/S2 niet gehaald) met v4 als
  bevroren beste exacte kandidaat voor E6-integratie.

## Open stappen (zie HANDOFF_E4_EN_VERDER_2026-08-15.md)

1. In-lus adoptie v4 via monkeypatch/subclass + G-E4-T1 meting @262100 met
   token-pariteit tegen het s5-anker.
2. Onafhankelijke verifier voor deze fase (nog niet geschreven).
