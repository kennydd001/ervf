# NERVF-0/1 — ERVF-replicatie op Nemotron 3.5 Lightning: baseline-lock en geometrie-audit

Datum: 2026-08-15
Namespace: **NERVF_NEMOTRON** (append-only, nieuw)
Status: **bevroren vóór uitvoering.**

## 0. Wat er al bestaat, en wat niet gedupliceerd wordt

Doorzocht op ERVF of een equivalent mechanisme in de Nemotron-lijn:

- `reports/streamq5_moe/P7_*` — de **bewezen Qwen3-30B-A3B ERVF-doorbraak**
  (48/48 verificatiepoorten, Q8-projectievlak 15,213 → 8,819 ms = 1,725×,
  Q5-expert 18,167 → 7,614 ms = 2,386×, end-to-end 20,029 → 30,113 tok/s).
  Kernelvorm: `WIDTH`-lane subwarp per rij, `VIRTUAL = 256/WIDTH` virtuele
  accumulatoren per lane, daarna reconstructie van exact dezelfde reductieboom.
  Gekozen breedte 16. **Dit is andermans lijn (streamq5) en blijft read-only.**
- P7's eigen vervolglijst noemt **"P8E tweede moderne MoE-checkpointreplicatie"**
  als openstaand punt. Deze fase ís die replicatie.
- In de Nemotron-runtime: **geen ERVF en geen equivalent aanwezig.**
  `gemv_nvfp4_rows` in `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py`
  gebruikt `const int row = blockIdx.x;` met 256 threads per rij, een
  block-brede `warp_sums`-reductie en `__syncthreads()` — structureel **exact de
  Qwen-vorm van vóór ERVF**.
- Gerelateerd maar niet hetzelfde: N7-C (gevectoriseerde `uchar4`-loads, al
  geadopteerd), S9 (blokgrootte-sweep — die varieerde de blockgrootte bij één
  rij per block, wat iets anders is dan meerdere rijen per block), X1's
  `gemm_nvfp4_rows_b` (batcht *activaties*, niet rijen).

Niets hiervan wordt herschreven. Alle nieuwe artefacten staan in
`reports|scripts/nervf_nemotron/`.

## 1. Waarom de hypothese hier plausibel is

| | Qwen vóór ERVF | Nemotron nu |
|---|---:|---:|
| raw scan | 357–361 GB/s | **338,4 GB/s** (N5, gemeten) |
| kritieke GEMV | 89–96 GB/s | **81,4 GB/s** (Y2-R1, gemeten) |
| verhouding | ~0,26 | **0,24** |
| kernelvorm | 1 block van 256 per rij | 1 block van 256 per rij |

Bovendien, en dit is nieuw: de huidige kernel stageert `x` per block in shared
memory. Bij 1856 rijen zijn dat 1856 blocks die elk 2688 floats lezen —
**19,96 MB activatieverkeer tegen 2,81 MB gewichtsbytes, 7,1×**. Zestien rijen
per block deelt die staging en snijdt dat verkeer 16×.

Dat is een mechanistische aanwijzing, geen bewijs. NERVF-1 moet het aantonen.

## 2. NERVF-0 — baseline-lock

Vastgelegd vóór enige kernelwijziging: model/checkpoint (`nemotron_3_5_lightning_v35`),
alle relevante bestandshashes (`runtime.py`, `fused_nvfp4.py`, `gpu_kernels.py`),
GPU-identiteit en klokken, contextconfiguratie, seeds, en de bestaande gemeten
baseline. De doorvoerbaseline wordt **niet opnieuw gemeten** — hij ligt vast in
`n7b_cached_decode.json` (27,574 / 25,523 / 21,794 / 18,358 tok/s) en in
`V35_GENERATION_ANCHOR.json` (2 × 64 tokens, bit-identiek anker).

De numerieke controle voor deze fase is de **kernel-uitvoer zelf**, want NERVF-1
en NERVF-2 raken alleen de GEMV.

## 3. NERVF-1 — reductie-geometrie-audit

Op één echt NVFP4 `up_proj`-record (1856 × 2688) en één echte activatie, alle
armen op dezelfde bytes, 200 aanroepen per sync (Y2-R1-protocol):

| arm | wat |
|---|---|
| `RAW_SCAN` | dezelfde gewichtsbytes puur sequentieel lezen |
| `ROW_PATTERN_SCAN` | het echte per-rij toegangspatroon, zonder MAC en zonder reductie |
| `DECODE_SCALE` | NVFP4-decode + blokschalen, zonder volledige dot-reductie |
| `FULL_GEMV` | de huidige productiekernel, ongewijzigd |
| `NO_X_STAGING` | de productiekernel met `x` uit global i.p.v. shared — isoleert de stagingkosten |

Gerapporteerd per arm: ms, effectieve GB/s, en uit de compiler:
registers/thread, statisch shared, en de theoretische occupancy.

**Geometrie-poort, vooraf vastgelegd (openen van NERVF-2 vereist beide):**

- **G-NERVF-1A:** `FULL_GEMV / RAW_SCAN ≤ 0,40` op tijd.
- **G-NERVF-1B:** minstens één van:
  - reductie + synchronisatie verklaart **≥ 25%** van de FULL_GEMV-tijd
    (gemeten als `FULL_GEMV − DECODE_SCALE`), of
  - een reductie-geometrie-microkernel voorspelt **≥ 1,25×**.

Faalt de poort, dan wordt ERVF **niet** geopend, wordt er geen breedte
nagetuned, en volgt een negatief rapport.

## 4. NERVF-2 — de ERVF-NVFP4-microkernel (alleen bij positieve poort)

Vooraf vastgelegde fysieke breedtes **w ∈ {4, 8, 16, 32}**, logische breedte
V = 256 zoals de referentie. Lane `L` houdt `256/w` gescheiden accumulatoren voor
de virtuele threads `tid = L + w·vi`, en de referentie-reductieboom wordt exact
gereconstrueerd:

1. de referentie doet eerst `__shfl_down` met offset 16 binnen een warp van 32;
   in de ERVF-afbeelding zijn dat twee virtuele accumulatoren **van dezelfde
   fysieke lane** — een lane-lokale optelling, geen shuffle;
2. de offsets 8/4/2/1 worden `__shfl_down_sync(..., w)` binnen de subwarp;
3. de acht warp-sommen worden in registers gecombineerd in exact de volgorde die
   de referentie via `warp_sums` en de tweede butterfly oplegt:
   `((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

Daarmee vervallen `warp_sums`, het shared-geheugen ervoor én `__syncthreads()`,
en delen `256/w` rijen één `x`-staging.

**Exactheidspoort G-NERVF-2C, hard:** de uitvoer is **bit-identiek** aan de
productiekernel, voor elke breedte, over: willekeurige codes, adversariële
magnitudes, **echte** activaties en gewichten, meerdere experts, vroege/midden/late
lagen, en een nul-zware ReLU²-invoer. Eén verschillend bit sluit die breedte uit.
Tolerantie wordt achteraf niet verruimd. Blijkt de productiekernel zelf niet
deterministisch, dan stopt de fase en wordt dát eerst gedocumenteerd.

**Snelheidspoorten:** primair **≥ 1,35×** op het projectievlak, sterk **≥ 1,75×**,
moonshot **≥ 2,0×** — allemaal alleen geldig bij volledig geslaagde exactheid.

Breedteselectie gebeurt op een validatieset van shapes; de gekozen breedte wordt
bevroren vóór de testmeting.

## 5. Wat deze fase niet doet

Geen wijziging aan gewichten, schalen, routing, activaties, MAC-toewijzing,
accumulatordtype, reductievolgorde of output-cast. Geen integratie in de runtime
(dat is NERVF-3). Geen combinatie met graph/gatherless/attention-winst — de
combinatieregel verbiedt het optellen van componentpercentages.

## 6. Artefacten

`scripts/nervf_nemotron/nervf01_geometry_audit.py` · `nervf0_baseline_lock.json` ·
`nervf1_geometry_audit.json` · `nervf2_ervf_microkernel.json` ·
`scripts/nervf_nemotron/nervf_independent_verify.py` ·
`nervf_independent_verification.json` · `NERVF_SHA256_MANIFEST.json` · rapport.

## 7. Claim boundary van dit document

Geen meting, geen resultaat.
