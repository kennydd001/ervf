# Beslisnota: single-stream uitputten of naar PRO-E100-BATCH?

Datum: 2026-08-16 · alle getallen gemeten, geen schattingen · schrijver: Claude
(sessie 2026-08-16, in-graph attributie)

**Kort antwoord: single-stream is NIET veelbelovend genoeg voor 100 tok/s. Ga
naar batch — maar met een fors lagere verwachting dan de eerste versie van deze
nota schreef.**

> ### ⚠️ STATUS 2026-08-16, na twee correcties op mijn eigen metingen
>
> De oorspronkelijke claim hieronder ("een orde beter", ×3,61 bij N=4) is
> **niet houdbaar**. Twee fouten, allebei gecorrigeerd:
> 1. de twee Mamba-shapes zijn **FP8**, niet BF16 (890 van 1686 MB/token);
> 2. veel belangrijker: de baseline was de **one-block-per-row**-kernel, maar
>    productie routeert deze shapes naar **ERVF-16**, dat **3,4-3,7× sneller**
>    is. Vrijwel de hele gemeten "batchwinst" was terrein terugwinnen dat ERVF
>    al had.
>
> **Eerlijke meting, batching gebouwd op de ERVF-geometrie, N=1 bitexact tegen
> productie-ERVF: ×1,38 bij N=2 en ×1,64 bij N=4** op 890 MB/token — niet ×3,5.
> Reden: ERVF haalt bij N=1 al **247-266 GB/s = 77% van het apparaatplafond**,
> dus er is nauwelijks bandbreedte over om terug te winnen, en bij N=4 wordt de
> kernel rekengebonden.
>
> Herziene projectie: **N=4 ≈ 13,0 ms ≈ 77 tok/s** (was 88). **Batch haalt de
> 100 bij N=4 dus niet, en bij N=8 is de marge onduidelijk.**
>
> Ondergrens-voorbehoud: mijn batched ERVF-kernel leest X uit global omdat N
> kopieën in shared de 48 KB-limiet overschrijden. Een getegelde versie die X
> in shared houdt kan beter zijn; hoeveel is **niet gemeten**.
>
> **DERDE CORRECTIE — de reikwijdte is groter dan gedacht.** Ik schreef hier dat
> `shared_up`/`routed_up` niet-ERVF zijn en daar ~3,6× zou blijven staan. Fout.
> `fused.gemv_into` doet `if self.use_ervf:` — **ERVF is de standaard** — en
> `up_kernels.run_batched` gebruikt dezelfde WIDTH-16-geometrie. Inventaris:
> **2031 van de 2048 MB/token (99,2%) gaat door een ERVF-kernel**; alleen de
> K/V-projecties (16,5 MB) niet.
>
> Daarmee geldt het gemeten ×1,64-bij-N=4 voor vrijwel de hele dense stroom.
> Herziene projectie: **N=4 ≈ 14,0 ms ≈ 71 tok/s**, N=8 ruwweg **~83 tok/s**.
>
> **Conclusie zoals het er nu staat: met de huidige kerneltechnologie haalt
> géén van beide routes de 100** — single-stream is hard begrensd op ~94, batch
> landt naar verwachting op 70-85.
>
> De diepere reden is opbouwend: *het succes van ERVF is precies wat de
> bovenkant van batching wegneemt.* Batching amortiseert gewichtslezingen, en
> dat werkt alleen als je bandbreedte verspilde. V4-V6 zijn al op 77% van het
> apparaatplafond. Dezelfde inefficiëntie kun je niet twee keer opeten.
>
> **De enige nog openstaande vraag die dit kan verleggen:** mijn batched
> ERVF-kernel leest X uit global (N kopieën in shared overschrijden 48 KB), dus
> ×1,64 is een **ondergrens**. Een K-getegelde variant die X in shared houdt is
> de laatste manier om de bovenkant te verhogen. **Dat ene getal beslist of
> batch richting 100 kan** — en het is niet gemeten.
>
> **Methodische regel die hieruit volgt:** vergelijk een kandidaat altijd met
> wat de runtime daadwerkelijk uitvoert voor díe shape. Check
> `BF16_ERVF_SHAPES` / `FP8_ERVF_SHAPES` vóór je een baseline kiest.

> **UPDATE dezelfde dag — de kernclaim is nu GEMETEN, niet meer beredeneerd.**
> `diag_batched_gemv_scaling.json`: Y[N,rows] = W·X[N,cols] op de zes echte
> shapes, koude rotatie, **N=1 bitexact tegen de productiekernel**, alle outputs
> eindig. MB-gewogen per-token-versnelling: **×1,94 bij N=2, ×3,61 bij N=4,
> ×5,10 bij N=8** — bij N=4 is dat 90% van perfecte schaling, en opvallend
> consistent over alle zes shapes (3,48-3,68×).
>
> **CORRECTIE (`diag_batched_gemv_fp8.json`):** de twee Mamba-shapes zijn FP8,
> niet BF16 (38.782.608 B voor 38,7M params = 1 byte/param). Zij dragen 890,6
> van de 1685,7 MB/token. Gecorrigeerd MB-gewogen: **×3,52 bij N=4, ×4,64 bij
> N=8** (was 3,61 / 5,10). Beide FP8-shapes bitexact en eindig.
>
> Toegepast op de in-graph componenttijden (projectie, geen meting):
> **N=4 ≈ 11,3 ms/token ≈ 88 tok/s · N=8 ≈ 9,5 ms/token ≈ 105 tok/s.**
> Bij N=8 komt het boven de 100 uit, waar single-stream op ~94 theoretisch
> maximum vastzit — maar met minder marge dan de eerste meting suggereerde.
>
> **Waarschuwing die daaruit volgt.** FP8 heeft de helft van BF16's bytes maar
> kost bij N=1 evenveel per token (297,6 vs 296,2 µs op dezelfde shape): die
> kernel is **niet bandbreedte-gebonden maar decode/reken-gebonden**, en
> verzadigt daarom eerder onder batching. Mamba's hoofdruimte onder batching is
> dus **kleiner dan de byte-boekhouding suggereert**. N=4 haalt 90% van perfecte
> schaling, N=8 nog maar 58% — weeg dat mee in `N_MAX`.
>
> VRAM is de eerste harde randvoorwaarde: per sequentie ~48 MB Mamba-state +
> ~12,6 MB KV bij ctx 4096, dus bij N=8 ~485 MB extra tegen ~605 MiB vrij.
> Het past, maar krap — dat bepaalt `N_MAX` in B0.

---

## 1. Wat single-stream nog kan opleveren — volledig uitgerekend

Basis in de gevangen graph: **21,24 ms = 47,08 tok/s** (drift 0,258 ms).
Alle marginalen hieronder zijn in-graph gemeten, alle armen bitexact.

| post | gemeten | behaald | vloer | hoofdruimte |
|---|---:|---:|---:|---:|
| MoE · gather | 3,849 | 16,6 GB/s (PCIe) | 2,47 | 1,38 |
| MoE · up_proj | 2,253 | 171,9 GB/s | 1,56 | 0,70 |
| MoE · shared_expert | 1,810 | 160,2 GB/s | 1,17 | 0,64 |
| MoE · down_masked | 1,372 | 46,6 GB/s | 0,26 | 1,11 |
| MoE · panel_scan + reduce + accumulate | 1,119 | — | — | ~1,0 |
| Mamba | 5,168 | 157,5 GB/s | 3,58 | 1,59 |
| attention | 2,479 | — | 1,13 | 1,35 |
| rest (lm_head, norms, embed, argmax) | ~3,7 | — | ~0,9 | ~2,8 |
| **totaal** | | | | **~10,6 ms** |

**Zelfs als je élke ms hoofdruimte pakt** — alle kernels perfect, PCIe volledig
verstopt — kom je op 21,24 − 10,6 = **10,6 ms ≈ 94 tok/s**. Onder de 100. En dat
is het theoretisch maximum van deze architectuur, niet een realistisch doel.

**Wat vandaag daadwerkelijk lukte, is het echte signaal.** Acht hypotheses
gebouwd en gemeten; de beste opbrengst was **−0,42 ms** (B3 in de graph) en de
rest was nul of negatief:

| poging | uitkomst |
|---|---|
| queue-starvation wegnemen (V12) | 0, weerlegd |
| H-SCALE schaalvlakken residentie (V13) | −0,37, onder eigen poort |
| B3 PCIe-overlap in de graph (V14-G) | **−0,42**, beste van de dag |
| gather-grid krimpen | negatief |
| gebatchte gather/down_masked (V15) | +0,04, neutraal |
| PV2-11 Q/K/V one-launch (V16) | +2,63, weerlegd |
| 32-lane ERVF-GEMV | neutraal |
| FP8-LUT-vrij | +25%, weerlegd |
| ERVF voor K/V-shape | 0,75×, weerlegd |

Negen pogingen, één winst van 0,42 ms. Om 10 ms te vinden heb je in dit tempo
~25 succesvolle ingrepen nodig terwijl de trefkans ~1 op 9 is. **Dat is geen
route naar 100.**

---

## 2. Waarom batch een orde meer hoofdruimte heeft

De hele kostenstructuur is **gewichtsbytes**, en die zijn voor élke sequentie in
een batch **identiek**. Uit de exacte byte-boekhouding (safetensors-headers):

| | MB/token | is per sequentie verschillend? |
|---|---:|---|
| Mamba | 892 | **nee** — zelfde gewichten |
| attention | 281 | **nee** |
| shared expert + gate | 290 | **nee** |
| lm_head | 198 | **nee** |
| **dense subtotaal** | **1661 (79%)** | **nee** |
| routed experts (up) | 387 | deels — de unie groeit sublineair |
| routed down (PCIe) | ~64 | idem |

**79% van al het verkeer is dense en wordt bij batch N één keer gelezen in
plaats van N keer.** En voor het routed deel is de overlap gemeten:
`diag_cross_sequence_union.py` vond bij N=16 een gemiddelde unie van **63,9 van
128 experts per laag = 66,6% van de no-overlap-baseline**.

Ruwe rekening bij N=4 (dense gedeeld, routed-unie ~2,5× die van N=1):

    (1661 + 387·2,5 + 64·2,5) / 4  =  ~695 MB per token
    695 MB / 249 GB/s              =  2,79 ms/token  =  ~358 tok/s theoretisch

Zelfs bij de **huidige** systeemefficiëntie (we halen ~45% van de vloer) is dat
~160 tok/s aggregate. **Er zit een factor 3-7 in, niet een factor 1,1.**

**Bijkomend, en het wordt vaak vergeten:** bij N>1 is een GEMV geen GEMV meer
maar een GEMM met kleine N. De rekenintensiteit per gewichtsbyte gaat ×N omhoog,
dus juist de kernels die nu op 157-172 GB/s bandbreedte-gebonden vastzitten
(Mamba, up_proj, shared_expert — samen 1569 MB/token) houden daarmee op
bandbreedte-gebonden te zijn. Dat is precies waar vandaag alle hoofdruimte zat
en waar single-stream er per definitie niet bij kan.

---

## 3. Waarom de bestaande N=2-prototypes dit NIET weerleggen

| opzet | aggregate tok/s |
|---|---:|
| N=2 naïef-eager | 31,66 |
| N=2 expliciete deling, Python | 11,23 |
| N=2 graph, private caches | 36,86 |
| single-stream | **47,08** |

Alle batch-prototypes zijn **slechter** dan single-stream — maar ze testen de
batch-hypothese helemaal niet. Ze **interleaven onafhankelijke volledige
passes** via state-swapping: elke sequentie leest alle 2048 MB opnieuw. Er wordt
niets gedeeld behalve toevallige cache-warmte. Ze meten de kosten van
interleaven, niet de baten van batchen. Dat is een belangrijk onderscheid en het
staat ook zo in `PATH_TO_100_TOKS.md`.

---

## 4. Aanbeveling

**Start PRO-E100-BATCH.** Volg `POST_V6_100TPS_PLAN.md`'s Path B, met deze
prioriteitsvolgorde die uit de metingen van vandaag volgt:

1. **B1 — dense shell eerst, en alléén dat.** Mamba + attention + shared expert
   + lm_head = **1661 van 2112 MB/token = 79%**, en het is het makkelijkste deel:
   geen routing, geen sparsity, geen LRU. Alleen een batchdimensie op de GEMV's
   (die daarmee GEMM's met kleine N worden). Poort: N onafhankelijke outputs
   gelijk aan N losse single-sequence V6-runs, 64 tokens. **Geen expert-deling
   in deze stap.**
2. **B2 — routed expert-unie** pas daarna, want dat is het moeilijke deel (de
   device-dedup + per-sequentie exacte maskers en reductievolgorde) en het is
   maar 21% van het verkeer.
3. **B3 — overlap en graph.** Het overlap-mechanisme is er al en is bitexact
   (`moe_dev_overlap.py`, in de graph −0,42 ms bij N=1).

**Wat er al ligt en herbruikbaar is:** de drift-stabiele meetharness (0,04-0,42
ms drift, waar de hele PRO-MAX V2-campagne op 1,9-3,2 ms strandde), de
`_recapture`-helper voor A/B's tussen gevangen graphs, de marginale
attributiemethode met bitexacte poort, de staging-race-fix in `step_graph`, en
de gemeten byte-boekhouding om elke claim tegen af te zetten.

**Eerlijke risico's, niet weggeschreven:** de runtime heeft **nul**
batch-ondersteuning (elke buffer is 1D, single-sequence) — dit is een
herontwerp van weken, niet een uitbreiding. VRAM is krap (207-605 MiB vrij), en
per-sequentie Mamba- en KV-state schaalt met N. En de eerste eerlijke
E100-claim vraagt ≥10.000 tokens, exacte outputs tegen losse referenties,
p50/p95/p99, fairness per sequentie en een thermische run — zie B4.

**Wat single-stream nog wél verdient:** B3 afmaken (het werkt en is bitexact) en
de gather van 16,6 naar dichter bij 25,9 GB/s brengen (1,38 ms hoofdruimte, de
grootste enkele post die er nog is). Dat is onderhoud, geen route naar 100.
