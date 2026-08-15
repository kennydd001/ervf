# NERVF-0/1 — ERVF-replicatie op Nemotron: baseline gelockt, geometrie half beslist

Datum: 2026-08-15
Namespace: `NERVF_NEMOTRON`
Status: **NERVF-1 INCONCLUSIVE. G-NERVF-1B slaagt robuust (43–47%), G-NERVF-1A is niet te beoordelen door een defect in mijn eigen referentie-arm. NERVF-2 is daarom NIET geopend.**
Terminal state: `nervf1_reduction_share_positive_raw_scan_arm_invalid`
Preregistratie: `NERVF_0_1_PREREGISTRATION_2026-08-15.md` (bevroren vóór uitvoering)

## 1. Wat er al bestond (geen duplicatie)

- De bewezen ERVF-doorbraak staat in **andermans lijn**,
  `reports/streamq5_moe/P7_*` — Qwen3-30B-A3B, 48/48 poorten, Q8-projectievlak
  1,725×, Q5-expert 2,386×, end-to-end 20,029 → 30,113 tok/s, breedte 16.
  Read-only gelaten.
- P7's eigen vervolglijst noemt **"P8E tweede moderne MoE-checkpointreplicatie"**
  als openstaand punt. Deze fase ís die replicatie.
- **In de Nemotron-runtime bestaat geen ERVF en geen equivalent.**
  `gemv_nvfp4_rows` geeft elke outputrij een volledig block van 256 threads en
  reduceert via shared memory met een `__syncthreads()` — structureel exact de
  Qwen-vorm van vóór ERVF. Bevestigd door inspectie.
- Wel aanwezig maar iets anders: N7-C (`uchar4`-loads, al geadopteerd), S9
  (blokgrootte-sweep bij **één** rij per block), X1 `gemm_nvfp4_rows_b` (batcht
  activaties, niet rijen).

## 2. NERVF-0 — baseline gelockt

`nervf0_baseline_lock.json`: model `nemotron_3_5_lightning_v35`, SHA-256 van
`runtime.py` / `fused_nvfp4.py` / `gpu_kernels.py` / de runner, GPU
`RTX PRO 2000 Blackwell, driver 596.58, 8151 MiB, SM-klok 3090 MHz`, cupy 14.1.1,
en de bevroren doorvoerbaseline (27,574 / 25,523 / 21,794 / 18,358 tok/s) plus het
generatie-anker. Geen tuning op testdata.

## 3. NERVF-1 — wat er wél en niet uit komt

Twee volledige runs, zelfde bytes, 200 aanroepen per sync:

| arm | run 1 | run 2 |
|---|---:|---:|
| `RAW_SCAN` | 9,77 µs | **51,67 µs** |
| `ROW_PATTERN_SCAN` | 10,46 µs | **107,13 µs** |
| `DECODE_SCALE` | 17,61 µs | 19,73 µs |
| `FULL_GEMV` | 33,47 µs | 34,74 µs |

**De twee referentie-armen zwaaien een factor 5 tot 10; de twee echte armen
bewegen 4–12%.** Dat is geen ruis maar een ontwerpfout: het expert-record is
2,81 MiB en past ruim in L2. `RAW_SCAN` en `ROW_PATTERN_SCAN` meten dus
L2-residentie, en die verschilt per run afhankelijk van wat er vóór hen draaide.
N5 vermeed dit indertijd bewust met een buffer van 256 MiB; ik heb die les hier
niet toegepast.

Gevolg: **G-NERVF-1A (`FULL_GEMV / RAW_SCAN ≤ 0,40` als bandbreedte-efficiëntie)
is met deze arm niet te beoordelen** en wordt niet als geslaagd of gefaald
gerapporteerd.

### Wat wel robuust is

| | run 1 | run 2 |
|---|---:|---:|
| `DECODE_SCALE` | 17,61 µs | 19,73 µs |
| `FULL_GEMV` | 33,47 µs | 34,74 µs |
| **reductie + synchronisatie** | **47,4%** | **43,2%** |

**G-NERVF-1B geslaagd, in beide runs ruim**: bijna de helft van de
FULL_GEMV-tijd zit ná de decode — in de dot-reductie, de shared-memory-boom en
de synchronisatie. Niet in geheugentoegang, niet in decode.

Dat is precies de Qwen-signatuur van vóór ERVF, en het is het deel van de
hypothese dat er voor ERVF toe doet.

### En één getal dat op zichzelf staat

De huidige kernel stageert `x` per block in shared memory. Bij 1856 rijen zijn
dat 1856 blocks die elk 2688 floats lezen: **19,0 MiB activatieverkeer tegen
2,7 MiB gewichtsbytes — 7,1×.** Zestien rijen per block deelt die staging en
snijdt dat 16×. Dit is onafhankelijk van de reductie-winst en is in de
Qwen-ERVF-analyse niet apart benoemd.

## 4. Waarom NERVF-2 niet geopend is

De preregistratie eist **beide** poorten voor opening. 1B slaagt, 1A is niet
beoordeelbaar. Ik open ERVF niet op één poort, en ik herformuleer 1A niet
achteraf naar iets wat de bestaande data wél haalt — dat is precies het soort
post-hoc dat de opdracht verbiedt ("Geen post-hoc widths tunen totdat iets
'werkt'").

De ERVF-microkernel is wél al geschreven en staat in de runner: `w ∈ {4,8,16,32}`,
`256/w` rijen per block, `256/w` gescheiden virtuele accumulatoren per lane, en de
referentie-reductieboom exact gereconstrueerd — inclusief de vondst dat de
eerste stap van de referentie (offset 16 binnen een warp van 32) in deze
afbeelding twee accumulatoren van **dezelfde fysieke lane** koppelt en dus een
lane-lokale optelling wordt in plaats van een shuffle. Hij draait pas als de
poort open is.

## 5. Wat de volgende stap exact is

Eén regel: **`RAW_SCAN` en `ROW_PATTERN_SCAN` moeten over een buffer groter dan
L2 lopen**, zoals N5's 256 MiB, door het expert-record te repliceren tot voorbij
de L2-grens of door over veel verschillende experts te scannen in plaats van
steeds hetzelfde record. Daarna is 1A in één run beslist en opent NERVF-2 of
niet.

Tot dan is de eerlijke stand: de reductie-geometrie-aanwijzing is **positief en
robuust**, de bandbreedte-aanwijzing is **ongemeten**.

## 6. Claim boundary

Microbenchmarks op één echt NVFP4 `up_proj`-record; effectieve GB/s telt alleen
het gewichtsrecord. Geen tokentijd, geen doorvoerresultaat, geen integratie. De
`RAW_SCAN`- en `ROW_PATTERN_SCAN`-cijfers worden **niet** als geldige referentie
gebruikt, om de reden in §3. Het aandeel reductie+synchronisatie is afgeleid als
`(FULL_GEMV − DECODE_SCALE) / FULL_GEMV` en is dus een verschil van twee
gemeten armen, geen directe meting van de reductie zelf.

## 7. Artefacten

`NERVF_0_1_PREREGISTRATION_2026-08-15.md` ·
`scripts/nervf_nemotron/nervf01_geometry_audit.py` ·
`nervf0_baseline_lock.json` · `nervf1_geometry_audit.json`
