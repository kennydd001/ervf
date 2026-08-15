# Na ERVF: de tweede inversie, en zes nieuwe hypotheses

**Datum:** 2026-08-12 · **Basis:** P7C-meting 33,208 ms test / 30,113 tok/s, bit-exact
**Type:** heranalyse + nieuwe hypotheses · **Status:** afleiding uit gemeten data

---

## 0. Eerst: mijn premisse was fout, mijn test was goed

`KERNEL_INVERSIE` stelde dat P6B naar een BF16-scratchbuffer dequantiseerde.
Broninspectie falsificeerde dat: `dequantized_weight_scratch_bytes: 0`, registers
werden al gebruikt. **Die hypothese was fout.**

Maar P7A was juist zo opgezet dat hij dat kon uitwijzen, met drie takken. De derde
tak vuurde:

| Bank | raw scan | row-pattern | GEMV | pattern/raw |
|---|---:|---:|---:|---:|
| Q8 | 356,81 GB/s | 96,43 | 91,16 | 27,03% |
| Q5 | 361,32 GB/s | 89,41 | 65,10 | 24,75% |

Het geheugenpad haalt **93–94% van de piek**. De fout zat in de launchgeometrie —
één 256-thread block per outputrij met acht blockbrede synchronisaties. ERVF loste
dat op en leverde 1,725× op de projecties en 2,386× op de experts, bit-exact.

Dat is precies waar een diagnose voor dient: de hypothese sneuvelde, de meting
wees de weg. Ik neem de correctie over.

---

## 1. De tweede inversie

| Term | vóór ERVF | na ERVF | aandeel nu |
|---|---:|---:|---:|
| **G — glue** | 14,193 ms (28,4%) | **14,961 ms** | **45,1%** |
| P — projecties Q8 | 15,360 ms (30,8%) | 8,819 ms | 26,6% |
| E — experts Q5 | 18,560 ms (37,2%) | 7,614 ms | 22,9% |
| β·m — transfer | 1,814 ms (3,6%) | 1,814 ms | 5,5% |
| **totaal** | 49,927 ms | **33,208 ms** | |

ERVF raakte E en P. **G is onaangeroerd gebleven en is nu de grootste term.**

Eerst was transfer de dominante term, en die is naar 5,5% gezakt. Nu is compute
gehalveerd, en is de glue — het deel dat niemand ooit heeft aangevallen — 45%.

---

## 2. De CUDA-graph-sluiting wordt verkeerd toegepast

P7A leverde nog een cijfer dat niet is gebruikt: `q8_noop_241` = 241 lege launches
in 1,6993 ms → **7,05 µs per launch**, nu gemeten in plaats van geschat.

| Populatie | launches | launchkosten | van de term | fractie |
|---|---:|---:|---:|---:|
| expertketen (P3B testte deze) | 192 | 1,35 ms | 17,11 ms | **7,9%** |
| glue (nooit getest) | ~772 | 5,44 ms | 14,96 ms | **36%** |

P3B mat graph/eager = 1,0076 op de **expertketen**. Bij 7,9% dispatch is dat exact
wat je verwacht: daar valt niets te winnen. En de no-opcontrole van P3B werd wél
9× goedkoper — dat bewijst dat graphs dispatch wegnemen.

De glue is een andere populatie: ~16 kernels per laag (RMSNorms, q/k-norm, RoPE,
KV-write, attention-score, softmax, value-reductie, twee residuals, routersoftmax,
top-k, gewogen som) plus finale norm, argmax en sampling. Vrijwel nul bytes, alleen
overhead. Daar bovenop komt de host-side PyTorch/ATen dispatch, die graphs volledig
elimineren.

> **De registryregel "CUDA Graphs zijn gesloten" geldt voor de expertketen en is
> nooit getest op de glue. Dat is nu de grootste post in het budget.**

---

## 3. De kernels hebben nog 2,4× over

| | bereikt na ERVF | raw scan | resterende marge |
|---|---:|---:|---:|
| experts Q5 | 152,5 GB/s (39,7%) | 361,3 (94%) | **2,37×** |
| projecties Q8 | 141,6 GB/s (36,9%) | 356,8 (93%) | **2,52×** |

E+P bij raw-scansnelheid: **6,67 ms** tegen 16,43 nu.

Concrete kandidaten voor die 2,4×, in volgorde van verwachte opbrengst:
- **Gevectoriseerde loads.** Een 16-lane subwarp leest een rij van 2048 Q5-codes =
  1280 B = 80 B per lane. Als dat per byte gebeurt zijn dat 80 loads; als `uint4`
  vijf. Dit is de eerste plek om te kijken.
- **Activatievector in shared memory**, één keer geladen, gebroadcast over alle
  16 rijen in het block in plaats van per rij opnieuw.
- **Meerdere outputrijen per lane** voor instruction-level parallelism — de
  latency van de MAC-keten verbergen achter meer onafhankelijk werk.
- **`cp.async` / TMA** op Blackwell voor het gewichtenstromen, zodat de load van
  rij *i+1* overlapt met de MAC van rij *i*.

---

## 4. De ladder vanaf hier

| Stap | totaal | tok/s test | tok/s rollout |
|---|---:|---:|---:|
| gemeten P7C ERVF | 33,21 ms | **30,1** | **20,9** |
| + graph/layer-fusie op de glue | 22,75 ms | 44,0 | 30,5 |
| + kernels naar 70% piek | 15,28 ms | 65,5 | 45,5 |
| + kernels naar raw-scan 93% | 13,03 ms | 76,8 | 53,3 |
| + CPU-misscompute (β·m → 0) | 11,22 ms | **89,1** | **61,9** |

---

## 5. Zes nieuwe hypotheses

### H-A · Layer-fused megakernel (P8-prioriteit 1)

**Hypothese.** Eén kernel per decoderlaag in plaats van ~19, met
`cooperative_groups::grid_group::sync()` voor de intra-laag afhankelijkheden
(q/k/v → RoPE → attentie → o → router → experts → residual). Launches per token:
~900 → ~100.

Dit is sterker dan CUDA Graphs, want graphs verlagen alleen de *host*-kosten;
een gefuseerde kernel verwijdert ook de per-kernel opstart- en afbouwkosten en
houdt de activatie in registers/shared memory tussen stappen.

**Poort.** G ≤ 6,0 ms op test (van 14,961). Bit-exactheid tegen P7C op alle
1.270 testlabels — dezelfde eis die ERVF haalde.
**Voorspelling:** G → 4,5–6,0 ms, totaal → 22,7–24,2 ms → **41–44 tok/s.**

**Risico.** De H2D voor cachemisses is host-gestuurd en past niet in de kernel.
Oplossing: houd de expert-H2D en de expertkernels buiten de fusie (die zijn al
efficiënt), en fuseer alleen de attentie- en normketen. Dat is ~16 van de 19
kernels per laag en het grootste deel van G.

### H-B · Het VRAM-allocatieprobleem als expliciete optimalisatie

**Waarom dit nieuw is.** Drie consumenten vechten om 7,96 GiB, en voor het eerst
heeft elk een **gemeten** kostenfunctie. Dat maakt het een oplosbaar
allocatieprobleem in plaats van een ontwerpkeuze.

| slots | cache | KV | max context | extra miss | extra β·m |
|---:|---:|---:|---:|---:|---:|
| 1640 (nu) | 4,62 GiB | 1,48 GiB | 16.165 | — | — |
| 1200 | 3,38 | 2,72 | 29.695 | +32,5 | +0,83 ms |
| 800 | 2,25 | 3,84 | 41.995 | +79,3 | +2,03 ms |
| 400 | 1,13 | 4,97 | 54.295 | +163,8 | +4,20 ms |

**Van 1640 naar 400 slots kost ~4,2 ms en levert 3,4× meer context.** Met INT8-KV
verdubbelt dat nogmaals. Dit is nooit onderzocht, omdat de expertcache altijd als
gegeven werd behandeld.

**Poort.** Maximaliseer context onder tok/s ≥ 15 en CE-neutraliteit. Rapporteer
de volledige Pareto-curve context × tok/s.

### H-C · De contextmuur — de grootste blinde vlek in het hele project

Alle P6B/P7C-metingen draaiden op contexten van **128 tokens**. De rollout op 512.
KV = 96,0 KiB per contextpositie:

| context | KV in VRAM | KV-leestijd/token | % van 33,21 ms |
|---:|---:|---:|---:|
| 128 (getest) | 0,012 GiB | 0,08 ms | 0,2% |
| 512 (rollout) | 0,047 GiB | 0,33 ms | 1,0% |
| 4.096 | 0,375 GiB | 2,64 ms | 8,0% |
| 16.384 | 1,500 GiB | 10,56 ms | 31,8% |
| **32.768** | **3,000 GiB** | **21,12 ms** | **63,6%** |
| 65.536 | 6,000 GiB | 42,25 ms | 127,2% |

**Bij 32K is de KV-lees alleen al meer dan de hele huidige token.** Er is dan een
vijfde term die in geen enkele meting voorkomt, en die alle conclusies over de
verdeling omgooit — net zoals transfer en glue dat eerder deden.

**Hypothese.** KV-kwantisatie naar INT8 halveert zowel de VRAM als de leestijd,
tegen een CE-kost die meetbaar is met exact dezelfde teacher-forced opzet als
P0C. Bij INT4 nogmaals.

**Poort.** Bij 8K en 32K context: relatieve CE ≤ 2% en tok/s ≥ 15. Meet
BF16/INT8/INT4-KV op dezelfde splits. Dit is de eerste stap die de claim van
"korte context" naar "bruikbare context" tilt.

### H-D · Prefill en TTFT — ongemeten en productkritisch

Bij een prompt van ≥1000 tokens is de expertunie praktisch de hele bank.
Minimale H2D: **18,65 GB / 26,341 GB/s = 708 ms.** Dat is de ondergrens van TTFT
voor een lange prompt, ongeacht rekenkracht.

Er is geen enkele TTFT-meting in het project. Voor een chatproduct is dat even
belangrijk als tok/s.

En prefill is structureel anders: batch = promptlengte, dus het is
**compute**-gebonden en zou GEMM in plaats van GEMV moeten gebruiken. ERVF's
geometrie is voor GEMV geoptimaliseerd. Een aparte prefill-kernel is
waarschijnlijk een grote, makkelijke winst.

**Poort.** TTFT bij promptlengtes 128 / 512 / 2048 / 4096, met prefill-tok/s
apart gerapporteerd. Voorspelling: 708 ms is de vloer bij ≥1000 tokens.

### H-E · Exactheidsbeperkte autotuning — de wetenschappelijke kern

ERVF's echte bijdrage is niet de subwarpbreedte. Het is dat de transformatie
**bewijsbaar niets aan de numeriek verandert**: 0 verschillende bits over
1.878.400 uitvoerelementen, 0 verschillende CE-waarden over 2.540 labels.

Dat ontkoppelt prestatiewerk volledig van kwaliteitsevaluatie. In vrijwel alle
kernelwerk verschuift de numeriek een beetje, waarna een volledige
kwaliteitsherhaling nodig is. Hier niet.

**Hypothese.** Formaliseer dit als een klasse: *exactheidsbewarende
kerneltransformaties* — herschikkingen van de launchgeometrie die de virtuele
threadindeling en de FP32-reductieboom emuleren. Bouw een **mechanische verifier**
die voor elke kandidaat bit-gelijkheid controleert, en zoek dan automatisch over
de geometrieruimte (subwarpbreedte, rijen per block, rijen per lane,
loadvectorbreedte, shared-memory-broadcast) met bit-exactheid als **harde
randvoorwaarde**.

Dat maakt van P8A (handmatig gekozen breedtes per projectietype) een
gestructureerde zoektocht, en het is publiceerbaar als methode los van dit model.

**Poort.** De autotuner vindt ≥ dezelfde configuratie als P7B handmatig, en
minstens één configuratie die sneller is bij gelijke bit-exactheid.

### H-F · Batch > 1 — de unie groeit sublineair

| batch | unieke experts/laag | ms/batch | **tok/s aggregaat** |
|---:|---:|---:|---:|
| 1 | 8,0 | 33,21 | 30,1 |
| 2 | 15,5 | 40,35 | 49,6 |
| 4 | 29,1 | 53,31 | 75,0 |
| 8 | 51,6 | 74,72 | **107,1** |
| 16 | 82,4 | 104,04 | 153,8 |

Bij batch 8 is de aggregate doorvoer 4,5× hoger, want P, G en β·m zijn per pass
en amortiseren volledig. Relevant zodra "meerdere parallelle agents" of
"meerdere gebruikers" een doel wordt.

Let op: dit is precies de omgekeerde conclusie van speculative decoding, dat op
dezelfde uniewiskunde stukliep. Het verschil is dat batch-tokens *allemaal*
worden geaccepteerd; speculatieve tokens niet.

---

## 6. Wat ik niet zou doen

- **De 32-GiB-limietproef vóór H-A.** Die is waardevol en hoort in de
  publicatie, maar hij verandert geen enkele meting — hij bevestigt alleen wat
  het geheugenlogboek al zegt (20,2 GB banken van 63,4 GiB). Eén dag werk, geen
  nieuwe kennis. Doe hem, maar niet als eerste.
- **Naar een groter model vóór H-C.** Een groter model met contexten van 128
  tokens meet niets nieuws. De contextmuur raakt élk model.
- **Nog een quantisatievariant.** E is nu 22,9% en de kernels hebben 2,4× over
  bij ongewijzigde bits.
- **De Q5-als-INT4-kern-plus-staart uit de eigen ideeënlijst.** Dat verlaagt
  bytes, terwijl de kernels op 40% van de bandbreedte draaien die ze al hebben.
  Eerst de 2,4× halen, dan pas minder bytes lezen.

---

## 7. Aanbevolen volgorde

| # | Stap | Kosten | Winst | Waarom |
|---|---|---|---|---|
| 1 | **H-A** layer-fused megakernel | 1–2 wk | 1,46× | 45% van het budget, nooit aangevallen |
| 2 | **H-C** KV-kwantisatie + contextsweep | 1 wk | opent 32K | grootste blinde vlek |
| 3 | **H-E** exactheidsverifier + autotuner | 1 wk | 1,3–2,4× | maakt alle kernelwerk goedkoop en veilig |
| 4 | **H-D** prefill/TTFT-meting + GEMM-kernel | 3 dgn | TTFT | productkritisch, ongemeten |
| 5 | **H-B** VRAM-allocatie-Pareto | 3 dgn | context | pure heranalyse, geen nieuwe kernels |
| 6 | 32-GiB-proef + thermische duurproef | 2 dgn | — | maakt de claim publiceerbaar |
| 7 | **H-F** batch >1 | 1 wk | 4,5× aggregaat | pas relevant bij meerdere gebruikers |
| 8 | Tweede MoE-familie (DeepSeek-V2-Lite) | 2 wk | generalisatie | pas na H-A/H-C |

---

## 8. De wetenschappelijke stand

Het project heeft nu twee resultaten die los van elkaar publiceerbaar zijn.

**Het systeemresultaat.** Een MoE van 30,5 miljard parameters met een expertbank
van 17,37 GiB draait volledig autoregressief op een laptop-GPU met 8 GB VRAM op
20,9 tok/s over 512 gegenereerde tokens, met ~1% CE-verlies, onafhankelijk
geverifieerd.

**Het methodische resultaat, en dat is het sterkere.** Een gemeten kostenverdeling
die twee keer is omgeklapt: eerst was transfer dominant (33,5% van de expertplane),
toen compute, nu de glue. Elke keer bleek de term die het veld optimaliseert niet
de bindende. Gedragen door dertien preregistreerde falsificaties én een
bit-exacte optimalisatie die bewijst dat de winst niet uit een kwaliteitsoffer
kwam.

Die tweede kant is zeldzaam. Het argument "optimaliseer eerst de meting, dan pas
het systeem" is makkelijk te beweren en bijna nooit met dit soort bewijs
onderbouwd.
