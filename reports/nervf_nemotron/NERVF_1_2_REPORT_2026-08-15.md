# NERVF-1/2 — ERVF repliceert op Nemotron: 1,936× bitexact op het projectievlak

Datum: 2026-08-15
Namespace: `NERVF_NEMOTRON`
Verdict: **Geslaagde tweede-modelreplicatie. Beide geometrie-poorten open en stabiel (bandbreedte-efficiëntie 0,322 ≤ 0,40; reductie+synchronisatie 46,1% ≥ 25%). De ERVF-microkernel is bij álle vier de breedtes bit-identiek aan de productiekernel, en w=16 haalt 1,936× — dezelfde breedte die Qwen's P7 koos. Primaire poort (1,35×) en sterke poort (1,75×) beide gehaald; moonshot (2,0×) net niet.**
Terminal state: `nervf2_ervf_replicated_bitexact_1_94x_width16`
Preregistratie: `NERVF_0_1_PREREGISTRATION_2026-08-15.md` (bevroren vóór uitvoering)
Vorig rapport: `NERVF_0_1_REPORT_2026-08-15.md` (INCONCLUSIVE — zie §1)

## 1. Eerst: de fout uit het vorige rapport is verholpen

Het eerste rapport moest NERVF-1 als **INCONCLUSIVE** afsluiten omdat mijn
referentie-armen over één record van 2,81 MiB liepen, dat in L2 past. `RAW_SCAN`
zwaaide daardoor tussen twee runs van 9,77 naar 51,67 µs terwijl de echte armen
4–12% bewogen.

Opgelost zoals N5 het indertijd deed: **alle armen** cyclen nu door een pool van
95 gerepliceerde records, 254 MiB, ruim boven de gemeten L2 van 32 MiB. Geen arm
is nog L2-resident, en alle armen zijn even koud.

Effect op de stabiliteit — `RAW_SCAN` over drie runs:

| | vóór (L2-warm) | ná (pool van 254 MiB) |
|---|---|---|
| run 1 | 9,77 µs | 12,42 µs |
| run 2 | 51,67 µs | 12,43 µs |
| run 3 | — | 12,43 µs |

Van een factor 5,3 spreiding naar 0,1%.

## 2. NERVF-1 — waar de bandbreedte instort

Alle armen dezelfde bytes, L2-koud, 200 aanroepen per sync:

| arm | µs | effectief | wat erbij komt |
|---|---:|---:|---|
| `RAW_SCAN` | 12,43 | 225,8 GB/s | — |
| `ROW_PATTERN_SCAN` | 17,08 | 164,3 GB/s | het echte per-rij toegangspatroon |
| `DECODE_SCALE` | 20,80 | 134,9 GB/s | NVFP4-decode + blokschalen |
| `FULL_GEMV` | 38,58 | 72,7 GB/s | de dot-reductie, shared-x-staging en sync |

| poort | eis | gemeten | |
|---|---|---:|:--:|
| **G-NERVF-1A** | `FULL_GEMV / RAW_SCAN` bandbreedte ≤ 0,40 | **0,322** | ✅ |
| **G-NERVF-1B** | reductie+sync ≥ 25% van FULL_GEMV | **46,1%** | ✅ |

De stap van 226 naar 73 GB/s is dus **niet** het geheugen (het toegangspatroon
kost 4,6 µs) en **niet** de decode (7,7 µs), maar de laatste stap: **17,8 µs, 46%
van de kernel, zit in de reductie, de shared-memory-boom en de synchronisatie.**

Dat is precies de Qwen-signatuur van vóór ERVF:

| | Qwen vóór ERVF | Nemotron nu |
|---|---:|---:|
| raw scan | 357–361 GB/s | 225,8 GB/s |
| kritieke GEMV | 89–96 GB/s | 72,7 GB/s |
| verhouding | ~0,26 | **0,322** |
| kernelvorm | 1 block van 256 per rij | 1 block van 256 per rij |

## 3. NERVF-2 — de ERVF-kernel, en de reconstructie die hem exact maakt

`WIDTH` fysieke lanes per rij, `256/WIDTH` rijen per block van 256 threads, en
lane `L` houdt `256/WIDTH` **gescheiden** accumulatoren voor de virtuele threads
`tid = L + WIDTH·vi`. Geen MAC verplaatst, geen accumulator vroegtijdig
samengevoegd.

De referentieboom wordt daarna exact herbouwd. Twee dingen die daarbij tellen:

1. **De offset-16-stap van de referentie wordt lane-lokaal.** In deze afbeelding
   zitten `tid` en `tid+16` bij `WIDTH ≤ 16` in dezelfde fysieke lane, als twee
   virtuele accumulatoren. Dat is een optelling in registers, geen shuffle.
2. **Die lane-lokale stappen moeten in butterfly-volgorde**, niet sequentieel.
   Dit is de fout die de eerste meetronde blootlegde: bij w=16 valt de butterfly
   toevallig samen met één stap en kwam hij exact uit, maar w=4 en w=8 gaven
   **72 van 72** mismatches. Na correctie — folds met stride `16/WIDTH`,
   `8/WIDTH`, … vóór de shuffles — zijn álle breedtes exact.

De acht warp-sommen worden ten slotte in registers gecombineerd in exact de
volgorde die de tweede butterfly van de referentie oplegt:
`((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

Daarmee vervallen `warp_sums`, het shared geheugen daarvoor en `__syncthreads()`,
en delen `256/WIDTH` rijen één `x`-staging.

### Uitkomst

| breedte | rijen/block | µs | effectief | speedup | bitexact | regs |
|---:|---:|---:|---:|---:|:--:|---:|
| 4 | 64 | 28,88 | 97,2 GB/s | 1,336× | ✅ 0/72 | 96 |
| 8 | 32 | 22,88 | 122,7 GB/s | 1,686× | ✅ 0/72 | 68 |
| **16** | **16** | **19,93** | **140,8 GB/s** | **1,936×** | ✅ 0/72 | 48 |
| 32 | 8 | 20,34 | 138,0 GB/s | 1,897× | ✅ 0/72 | 40 |

Exactheid getest over 3 lagen × 3 experts × 4 activatieregimes (willekeurig,
adversariële magnitudes over 16 ordes, nul-zwaar, dicht) × 2 ReLU²-instellingen
= 72 gevallen per breedte, telkens tegen de productiekernel.

| poort | eis | gemeten | |
|---|---|---:|:--:|
| **G-NERVF-2C exact** | bit-identiek, hard | 0 mismatches, alle breedtes | ✅ |
| **G-NERVF-2S primair** | ≥ 1,35× | **1,936×** | ✅ |
| sterk | ≥ 1,75× | 1,936× | ✅ |
| moonshot | ≥ 2,0× | 1,936× | ❌ net niet |

**De gekozen breedte is 16 — dezelfde die Qwen's P7 selecteerde.** Dat is een
onafhankelijke bevestiging op een architectonisch ander model, andere
quantisatie (NVFP4 tegen Q5/Q8) en andere shape.

## 4. Op de doorbraakladder

**LEVEL 2 gehaald** (≥1,35× projectie, exact). LEVEL 3 (≥1,5× volledig
expertpad) vraagt ook de down-projectie en is NERVF-3.

Ter vergelijking met de eerste replicatie:

| | Qwen P7 | Nemotron NERVF |
|---|---:|---:|
| projectievlak | 1,725× (Q8) / 2,386× (Q5) | **1,936×** |
| gekozen breedte | 16 | **16** |
| bitexact | ja | ja |

## 5. Wat dit nog niet is

Dit is het **projectievlak**, geïsoleerd. Geen tokentijd, geen tok/s, geen
integratie in de runtime — dat is NERVF-3. De winst mag niet bij die van
attention-v4, graph of gatherless worden opgeteld; de combinatieregel eist een
nieuwe A/B per combinatie.

## 6. Claim boundary

Microbenchmarks op echte NVFP4 `up_proj`-records uit dit checkpoint, alle armen
op dezelfde bytes en allemaal L2-koud via een pool van 254 MiB. Effectieve GB/s
telt alleen het gewichtsrecord. Bitexactheid is gemeten tegen de productiekernel
over 72 gevallen per breedte en is een **harde** poort, niet een tolerantie. Geen
tokentijd, geen doorvoerresultaat, geen integratie, geen kwaliteitsclaim. De
vergelijking met Qwen P7 is een vergelijking van twee losse
projectievlak-metingen op twee modellen, geen gedeelde benchmark.

## 7. Artefacten

`NERVF_0_1_PREREGISTRATION_2026-08-15.md` ·
`scripts/nervf_nemotron/nervf01_geometry_audit.py` ·
`nervf0_baseline_lock.json` · `nervf1_geometry_audit.json` ·
`nervf2_ervf_microkernel.json` ·
`scripts/nervf_nemotron/nervf_independent_verify.py` (46/46, `VERIFIED`) ·
`nervf_independent_verification.json` · `NERVF_SHA256_MANIFEST.json`
