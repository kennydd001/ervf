# Overdracht aan Kimi — 3.5 Lightning is live, alle poorten gehaald

Datum: 2026-08-15
Van: Claude (deze sessie) · Voor: Kimi (S-lijn)
Status: **model gewisseld, runtime draait, 4 van 4 opdracht-poorten gehaald**

## Eerst: je S-werk klopte en is volledig overgenomen

Ik heb je S1–S7 gelezen en gebruikt. Twee dingen om te weten:

1. **Ik begon per ongeluk je GQA-kernel opnieuw te bouwen** voordat ik zag dat
   `attn_decode_warp_fp8_gqa` al bestond en gewired was. Mijn duplicaat is
   verwijderd (`_cleanup_duplicate_gqa.py`); jouw implementatie is wat draait.
2. **Je S4-conclusie is heropend — maar je had gelijk.** Je schreef "0 van
   24.147 keys, geen draft-gewichten". Dat klopte **voor Nemotron 3 Nano**. Zie
   hieronder.

## Het model was verkeerd — mijn fout, niet die van jou

De hele lijn draaide op `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. Het doel van de
opdracht is **3.5 Lightning**, en die weights waren publiek sinds
2026-08-13T18:36:59Z — ~15 uur vóór ik begon. Mijn `N0R_IDENTITY_REFRESH` was
precies de fase die dat had moeten vangen; ik controleerde de NIM-catalogus en
de gepinde repo, maar zocht nooit op HF naar een Lightning-checkpoint.
Correctie: `N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md`.

Model staat nu in `models/nemotron_3_5_lightning_v35` (52 shards, 20,08 GiB).
Zet `LS_MODEL_DIR=nemotron_3_5_lightning_v35` om ermee te draaien.

## N2R: drie gewichtsformaten, niet twee

Dit is het belangrijkste technische feit. De inline `quantization_config`
(8-bit float) misleidt — die geldt alleen voor Mamba.

| module | formaat | bewijs uit tensor-entries |
|---|---|---|
| routed experts, shared, **lm_head** | **NVFP4** | U8 `[1856,1344]` + F8_E4M3 `[1856,168]` + F32 `weight_scale_2` |
| Mamba `in_proj`/`out_proj` | **FP8 per-tensor** | F8_E4M3 `[10304,2688]`, één byte per gewicht, + F32 scalar |
| attention q/k/v/o, norms | BF16 | — |

Verschillen met Nano: expert-record 5.612.552 B (was 5.612.560; **geen
`input_scale`**), `layers_block_type`-lijst i.p.v.
`hybrid_override_pattern`-string (patroon zelf identiek: 23 M / 23 E / 6 ✳),
context 1.048.576 i.p.v. 262.144, en **270 MTP-tensors** (2,49 GiB, BF16).

## Wat ik heb gebouwd

- `loader.pattern_string()` — beide config-spellingen
- `loader.quant_kind()` — drieweg nvfp4 / fp8_tensor / bf16
- `gpu_kernels.gemv_fp8_tensor` — nieuwe FP8-per-tensor GEMV, `uchar4`-loads +
  gedeelde 256-LUT (arithmetische E4M3-decode maakte het FP8-KV-pad
  compute-bound; die meting is gerespecteerd)
- `runtime` dispatcht nu op formaat i.p.v. een boolean; `lm_head` gaat via de
  NVFP4-fused-GEMV

## Resultaat

| context | Nano beste | **3.5 Lightning** | winst |
|---:|---:|---:|---:|
| 0 | 20,676 | **27,743** | +34,2% |
| 32.768 | 19,939 | **26,200** | +31,4% |
| 131.072 | 17,202 | **21,699** | +26,1% |
| 262.100 | 15,181 | **18,424** | +21,4% |

Config: capacity 72, embed op host, FP8 KV, jouw GQA-kernel, jouw S5 masked
down_proj. Hitrate **80,4%**, shell 2,521 GiB, cache 4,328 GiB, VRAM 0,000 vrij.
Generatie coherent (`' humans started counting on their fingers. The first
counting device was the abacus, inven'`).

| poort | vereist | gemeten | |
|---|---:|---:|:--:|
| 4K minimum | 20 | ~27,7 | ✅ |
| **4K primary** | **25** | **27,743** | ✅ |
| 128K minimum | 10 | 21,699 | ✅ |
| **128K primary** | **15** | **21,699** | ✅ |
| 256K doel (gebruiker) | 30 | 18,424 | ❌ |

**Alle vier de opdracht-poorten gehaald.** De 30 @256K is een doel van de
gebruiker, geen opdracht-poort.

Winst komt uit twee dingen die de uitgever gaf, niet uit onze optimalisatie:
`lm_head` 704 → 198 MB (S6 mat die term op 5,965 ms), en FP8-Mamba. De kleinere
shell gaf meteen meer cache-slots.

## Voor jou: S4 opnieuw, en het is nu de sterkste hypothese

`num_nextn_predict_layers: 1`, `mtp_layers_block_type: ['attention','moe']`,
270 tensors, 2.670.652.160 B. Speculatief decoderen is de **enige** techniek die
*bytes per token* verlaagt in plaats van tijd per byte — precies wat we nodig
hebben nu alles transfer-bound is.

Kanttekening die je meteen moet meten: de MTP-experts zijn **BF16**
(`up_proj [1856,2688]` = 9.977.856 B), dus 3,5× groter per expert dan de
NVFP4-routed experts. Een MTP-forward is daarmee duurder per token dan een
gewone. De vraag is of de acceptatiegraad dat terugverdient.

Suggestie voor de meting vóór je bouwt: draaikosten van één MTP-forward, en de
acceptatiegraad op echte prompts. Pas als `accept × besparing > MTP-kosten` is
bouwen zinvol.

## Open punt dat ik niet heb opgelost

Je S7-mechanismecheck toonde **10,66×** amplificatie op kernel-niveau
(heads=32 5,8463 ms vs heads=2 0,5483 ms @244,79 GB/s = roofline). De
gegroepeerde kernel levert end-to-end veel minder dan die factor voorspelt. Dat
gat is **niet verklaard**. Een componentmeting van de gegroepeerde kernel in de
echte runtime (zoals S6) zou dat moeten uitwijzen — ik heb het niet gedaan.

## Regels ongewijzigd

Alleen schrijven in de `lightningstream_nemotron`-paden. Protected na elke fase:
**0 modified / 0 removed** (toegevoegd = Codex + jij). GPU delen via
`nvidia-smi --query-compute-apps`. Eén variabele per meting — ik heb daar zelf
tegen gezondigd (H1+H2 tegelijk in N8) en dat kostte een extra ronde.

## Bestanden

`N2R_V35_LAYOUT_REPORT_2026-08-14.md` · `n2r_v35_layout.json` ·
`N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md` · `n7b_cached_decode.json`
(laatste meting) · `_add_fp8_gemv.py` · `_wire_v35_runtime.py` ·
`_cleanup_duplicate_gqa.py`
