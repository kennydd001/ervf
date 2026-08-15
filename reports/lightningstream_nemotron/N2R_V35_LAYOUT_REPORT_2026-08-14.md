# N2R — 3.5 Lightning layout-adjudicatie

Datum: 2026-08-14
Verdict: **NVFP4 bevestigd voor experts; drie formaten in plaats van twee; MTP-gewichten aanwezig. De runtime draagt grotendeels over, maar niet ongewijzigd.**
Terminal state: `n2r_v35_layout_adjudicated`

Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, sha `6dbbd757…`
Lokaal: `models/nemotron_3_5_lightning_v35`, 52 shards, 20,08 GiB, 18.487 tensors.

## De beslissende vraag: NVFP4 of FP8?

**Beide.** De inline `quantization_config` (8-bit float) misleidde: die geldt
alleen voor de Mamba-projecties. Gemeten uit de echte tensor-entries:

| module | formaat | bewijs |
|---|---|---|
| routed experts, shared experts, **lm_head** | **NVFP4** | U8 `[1856,1344]` (1344 = 2688/2) + F8_E4M3 `[1856,168]` (168 = 2688/16) + F32 `weight_scale_2` |
| Mamba `in_proj` / `out_proj` | **FP8 per-tensor** | F8_E4M3 `[10304,2688]` — één byte per gewicht, niet gepakt — + F32 scalar `weight_scale` + F32 `input_scale` |
| attention q/k/v/o, norms | BF16 | `[4096,2688]` BF16 |

## Verschillen met Nemotron 3 Nano

| | Nano | 3.5 Lightning |
|---|---:|---:|
| laagpatroon | `hybrid_override_pattern` (string) | `layers_block_type` (lijst) |
| patroon zelf | `MEMEM*EMEMEM*…` | **identiek**, 23 M / 23 E / 6 ✳ |
| expert-record | 5.612.560 B | **5.612.552 B** |
| expert-velden | codes, scale, `weight_scale_2`, `input_scale` | idem **zonder `input_scale`** (−8 B) |
| routed bucket | 16.523.376.640 | 16.523.353.088 (= −2944 × 8) |
| shared bucket | 258.177.392 | 258.177.208 (= −23 × 8) |
| trunk/other | 2.558.227.600 | **2.085.387.040** |
| `lm_head` | BF16, 704.643.072 B | **NVFP4, 198.180.864 B** |
| Mamba in/out_proj | NVFP4 of BF16 | **FP8 per-tensor** |
| context | 262.144 | **1.048.576** |
| MTP | afwezig (S4: 0 van 24.147) | **270 tensors, 2.670.652.160 B, BF16** |

## Wat dit betekent

**Overdraagbaar zonder wijziging:** de NVFP4-codec, `uchar4`-loads, de fused
decode+GEMV, de kolom-selectieve down_proj van S5, warp-per-position attention,
FP8 KV, Kimi's GQA-kernel, de LRU-cache, de bankbouwer. Alle vormen en
recordgroottes zijn op 8 byte na gelijk, en het expert-record leest geen
`input_scale`.

**Vereist werk:**
1. `layers_block_type`-spelling — **gedaan** (`pattern_string()`, patroon
   geverifieerd identiek).
2. Drieweg `quant_kind()` — **gedaan** (nvfp4 / fp8_tensor / bf16).
3. **FP8-per-tensor GEMV-kernel** voor de 23 Mamba `in_proj`/`out_proj`. Bestaat
   nog niet; de runtime faalt zonder.
4. `lm_head` via de NVFP4-fused-GEMV in plaats van `gemv_bf16`.

**Cadeau van de uitgever:** `lm_head` van 704 MB BF16 naar 198 MB NVFP4 is 3,5×
minder verkeer op een term die S6 op 5,965 ms mat bij 262K. Puur uit het
formaat, zonder eigen werk.

**S4 heropend:** Kimi's census sloot speculatief decoderen omdat Nano geen
draft-gewichten had (0 van 24.147 keys). 3.5 Lightning heeft
`num_nextn_predict_layers: 1`, `mtp_layers_block_type: ['attention','moe']` en
270 MTP-tensors. Die conclusie gold voor Nano en moet hier opnieuw — het is de
enige techniek die *bytes per token* verlaagt in plaats van tijd per byte.
Kanttekening: de MTP-experts zijn BF16 (9.977.856 B per matrix), dus 3,5× groter
per expert dan de NVFP4-routed experts.

## Claim boundary

Header- en index-adjudicatie van het lokale 3.5 Lightning-checkpoint. Geen
payload gedecodeerd, geen module uitgevoerd, geen tok/s-, kwaliteits- of
contextclaim. Dat de runtime overdraagt is een gevolgtrekking uit vormen, geen
meting — N3R moet één module tegen de officiële code van dít checkpoint zetten
vóór er iets gemeten wordt.

## Artefacten

- `scripts/lightningstream_nemotron/n2r_v35_layout.py` · `n2r_v35_layout.json`
- `src/moe_lab/lightningstream_nemotron/loader.py` (`pattern_string`, `quant_kind`)
