# Staan we op de rand? Ja — maar niet op de rand die je denkt

**Datum:** 2026-08-12 · **Basis:** breakthrough-fase (GaugePack, ERGV, TierFlow, PORT80B)
**Type:** heranalyse van gemeten data · **Status:** afleiding, geen nieuwe run

---

## 0. Mijn P18A-idee is dood, en de reden is belangrijk

Ik noemde groepsbehoudende pruning "de belangrijkste regel in dit document" en
"de grootste hefboom". Dat bouwde volledig op P9B's kwaliteitsuitslag
(+1,478% / −0,478%).

**Die uitslag was een bug.** `weight[boolean_mask].zero_()` muteert een
advanced-indexing kopie, niet de parameter. P9B mat de gewone Q5-baseline. De
mutatie-audit bewees ongewijzigde gewichtshashes en bitexacte expertoutput.

Correcte uitkomsten:

| kandidaat | relatieve CE | top-1 |
|---|---:|---:|
| bevroren P9B-masker, 50% behouden | **+47,804%** | 60,866% |
| 64/128 per oorspronkelijke groep, 50% behouden | **+47,186%** | 60,866% |
| 25% pruning, 75% behouden | **+22,846%** | 74,803% |

Mijn layoutontwerp was correct — bit-exact, groepsidentiteit bewaard, 0,502363
byteratio bevestigd. **De premisse was fout.** Een perfecte implementatie van een
onhoudbaar kwaliteitsresultaat blijft waardeloos, en dat GaugePack de codec- en
kernelfase heeft stopgezet vóór een misleidende snelheidsclaim is precies goed.

### Wat er wél uit volgt, en dat is nieuw

Twee meetpunten geven een schaal:

```
50% pruning -> +47,804%
25% pruning -> +22,846%     ratio 2,09 bij een 2x kleinere pruningfractie
```

Vrijwel lineair in de weggesneden fractie. Extrapolatie: 10% → ~9%, 5% → ~4,6%,
**2% → ~1,8%** — dat is het hele kwaliteitsbudget voor een pruning van twee
procent.

> **Pruning is dood voor MoE-experts zonder hertraining, op elke fractie.**

En dat is een echt inhoudelijk resultaat. Wanda/SparseGPT halen 50% op dense
LLM's voor een paar procent perplexity. Op MoE-experts kost dezelfde 50%
+47,8%. De verklaring is structureel: een dense FFN moet álle inputs bedienen en
draagt daardoor enorme redundantie; een expert die 6,25% van de tokens ziet is
al gespecialiseerd en heeft die redundantie niet. **Expertniveau-redundantie is
fundamenteel lager dan dense-FFN-redundantie.** Dat is publiceerbaar, met drie
meetpunten en een full-depth protocol eronder.

---

## 1. De vondst: PORT80B faalt op 1,99×, niet op fysica

Dit is het belangrijkste getal in de hele nieuwe bundel.

| | ms | effectieve doorvoer | % van gemeten PCIe |
|---|---:|---:|---:|
| zero-cache mean | 65,530 | 14,85 GB/s | 56,4% |
| zero-cache p50 | 63,034 | 15,44 GB/s | 58,6% |
| zero-cache **p95** | **73,544** | 13,23 GB/s | **50,2%** |

Actieve set: 480 records × 2.027.520 B = **973,2 MB/token**.

```
fysieke vloer bij volle PCIe:  0,973 GB / 26,341 GB/s = 36,95 ms
de bevroren poort:             <= 45 ms
36,95 < 45  ->  DE POORT IS HAALBAAR
```

**De gate faalt op een efficiëntieprobleem, niet op een bandbreedtemuur.** Er
ligt 1,99× klaar, en het gat is precies te lokaliseren:

| pad | per record | 480 records |
|---|---:|---:|
| memcpy uit page cache @48,1 GB/s | 42,2 µs | 20,3 ms |
| H2D @26,341 GB/s | 77,0 µs | 36,95 ms |
| perfect gepipelined (8 vensters) | max = 77,0 µs | **36,95 ms** |
| volledig serieel per record | 119,2 µs | 57,18 ms |
| **gemeten p50** | **131,3 µs** | **63,03 ms** |

De meting is **12 µs per record trager dan volledig serieel**. Er zit dus iets
in dat noch memcpy noch H2D is.

---

## 2. De oorzaak staat in jullie eigen preflight-JSON

```
totaal fysiek RAM     63,43 GiB
in gebruik bij start  21,75 GiB
beschikbaar           41,68 GiB
bank                  46,50 GiB
                      ----------
tekort                 4,82 GiB  (10,4% van de bank)
```

**De bank past niet in het beschikbare RAM.** Peak working set was 46,845 GiB —
het proces duwde alles anders eruit en paginde alsnog. `Page Reads/sec` max
7.759, en de p50 van nul laat zien dat het episodisch is: precies het profiel van
een working set die net niet past.

Een record is **495 pagina's van 4 KiB**. Bij 10,4% niet-residentie raakt
praktisch elk record minstens één niet-residente pagina, en elke harde fault is
80–150 µs NVMe-latentie **op het kritieke pad**. Dat verklaart de 12 µs/record
bovenop serieel én de p99-uitschieter van 121 ms en de max van 1.358 ms.

Het rapport zegt: *"96 GB RAM is allowed for a controlled comparison but is not
proven to solve the bottleneck."* Terecht voorzichtig — maar de aritmetiek wijst
één kant op.

---

## 3. De beslissende test kost nul euro en een halve dag

Geen aankoop nodig. **Beperk de synthetische routegenerator tot een prefix van de
bank**, zodat de working set wél past:

| routebereik | working set | past in 41,68 GiB |
|---:|---:|---|
| eerste 60% | 27,90 GiB | ja, ruim |
| eerste 70% | 32,55 GiB | ja |
| eerste 80% | 37,20 GiB | ja, krap |
| 100% (nu) | 46,50 GiB | **nee** |

Zelfde code, zelfde 480 records per token, zelfde acht vensters, zelfde
H2D-pad. Enige wijziging: één parameter in de routegenerator.

- **p95 zakt richting 37 ms** → paging bevestigd; 96 GB lost het op en de gate
  wordt gehaald.
- **p95 blijft rond 73 ms** → het dispatchpad is de oorzaak; RAM kopen helpt
  niet en de vensterketen moet herzien.

Dit is exact de discriminatie die het rapport zelf in vier punten vraagt, maar
dan met één run en zonder hardware-aankoop. **Dit zou ik als eerste doen.**

---

## 4. Wat 80B dan wordt

| Component | ms |
|---|---:|
| transfer bij volle PCIe, zero cache | 36,95 |
| expertcompute bij ERVF-tempo (1,510 Gweight, 152,5 GB/s) | 6,34 |
| attentie + projecties + rest (naar analogie Qwen30) | 38,12 |
| **totaal** | **81,41 ms → 12,3 tok/s** |
| met de huidige paging | 118,01 ms → 8,5 tok/s |

Twee dingen vallen op.

**De actieve gewichten zijn vergelijkbaar met Qwen30.** 1,510 Gweight tegen
1,812. Het 80B-model is per token niet zwaarder — het is alleen veel groter op
schijf.

**De cache doet vrijwel niets.** 2.420 slots van 24.624 experts = 9,8%
residentie, gemeten hit 9,38%. Met 512 experts per laag en top-8 is de routing
veel ijler dan Qwen30's 8-van-128. Alle cachewerk uit de HERA/DHERA-campagnes is
op deze architectuur structureel waardeloos.

**Gevolg: bij 80B is transfer 45% van het budget en PCIe is een harde muur.**
973 MB/token bij 26,341 GB/s = 27 tok/s plafond, ongeacht alle kernels.

---

## 5. De iGPU is nu geen bijzaak meer, maar de enige uitweg

Bij Qwen30 was de iGPU een leuke optie. Bij 80B is hij de enige route die de
PCIe-muur omzeilt, want de Arc Pro 140T leest dezelfde gemapte bank rechtstreeks
uit DDR5:

| iGPU levert naast PCIe | samen | transfer/token | winst |
|---:|---:|---:|---:|
| 24 GB/s | 50,3 GB/s | 19,3 ms | 1,91× |
| 35 GB/s | 61,3 GB/s | 15,9 ms | 2,33× |
| 50 GB/s | 76,3 GB/s | 12,7 ms | 2,90× |

Het anker uit jullie eigen `p11b`: de Arc Pro 140T deed batch-8 in 1,315 ms tegen
CPU 2,371 en NPU 2,149 — **57,4 GB/s**, en hij was niet eens het onderwerp van de
test. P11A falsificeerde één CPU-kernel; dat is geen uitspraak over de iGPU.

Eerlijke waarschuwing: de iGPU en de dGPU-staging delen dezelfde DDR5-bus. Bij
50 GB/s totaal en 26,3 voor de dGPU-staging blijft er ~24 over. Vandaar dat de
eerste rij het realistische scenario is: **1,9×.**

---

## 6. ERGV is het echte resultaat, en het wordt onderschat

| bank | p50-ratio vs handmatige P7 | winst |
|---|---:|---:|
| Q8 | 0,8599 | 16,30% |
| Q5 | 0,9271 | 7,87% |

Tegen de later handgetunede N1C-graaf is het pariteit (0,9980 / 1,00007). Het
rapport noemt dat terecht "geen nieuwe end-to-end versnelling".

Maar dat is de verkeerde meetlat. **De compiler haalde mechanisch het beste
handwerk in, met 63/63 onafhankelijke checks, 2.680/2.680 CPU-bitchecks en 6/6
afgewezen semantische mutaties.** Dat betekent dat de volgende kernel, het
volgende model en de volgende GPU géén weken handwerk meer kosten — en dat de
verificatie automatisch is.

Dat is precies de as die dit project uniek maakt. Elke winst tot nu toe —
ERVF 2,39×, EVT-PM 7,47×, N1B vectorloads 1,126×, N1C 1,184×/1,070×, Concat-QKV
1,134× — kwam uit een **exactheidsbewarende transformatie**: nul bitverschillen
bewezen, geen kwaliteitsoffer. ERGV automatiseert die klasse.

De verdedigbare grens die het prior-artrapport trekt is ook de juiste:
*behoud van een gekozen bron-reductie-DAG onder een gewijzigde fysieke topologie*
— niet "eerste deterministische reductie".

---

## 7. Het antwoord op de vraag

**Nee, niet op de rand van een LLM-wereldrecord.** De 1–4% component-integraties
die de 2%-poort net missen zijn het klassieke signaal van een uitgeputte
zoekruimte. `breakthrough_claim_allowed: false` is de juiste status.

**Ja, op de rand van drie andere dingen, en die zijn verdedigbaarder:**

1. **Een gemeten 1,99× die klaarligt in PORT80B**, met een gratis test die
   uitwijst of RAM of dispatch de oorzaak is. Dat is het enige harde,
   onaangeroerde getal in de hele bundel.

2. **Een werkende exacte-reductiecompiler** die handgetuned werk mechanisch
   evenaart en zichzelf verifieert. Dat is een systems/compiler-bijdrage die los
   staat van dit model en deze laptop.

3. **Een negatief corpus dat zijn eigen fouten vangt.** Drie zelfcorrecties op
   rij: de P9B no-op bug, mijn glue-misattributie, de N4B BF16-afrondingsfout.
   Elk gevonden door het eigen protocol, elk bewaard in plaats van weggepoetst.
   Vrijwel geen gepubliceerd systemswerk heeft een mechanisme dat een
   `weight[mask].zero_()` no-op vangt. **Dat is het sterkste methodische bewijs
   in het hele project**, en het is precies wat de negatieve resultaten
   geloofwaardig maakt.

---

## 8. Voorgestelde volgorde

| # | Stap | Kosten | Waarom |
|---|---|---|---|
| 1 | **PORT80B route-prefix-test** (60/70/80/100%) | ½ dag | beslist 1,99× vs RAM-aankoop, gratis |
| 2 | Bij bevestigde paging: dezelfde run op 96 GB | €200 | dan is de gate gehaald en 80B loopt |
| 3 | **iGPU-expertkernel** via Level Zero/SYCL, bit-exact getoetst | 1–2 wk | enige route om de PCIe-muur te omzeilen, 1,9× |
| 4 | ERGV op tweede GPU + gematchte publieke kernels | 1–2 wk | maakt de compilerclaim extern verdedigbaar |
| 5 | Attentie naar ≥30% van piek (EVT-PM staat op 8,1%) | 1 wk | grootste resterende kernelmarge |
| 6 | TierFlow alleen met vast klein trainingsbudget | apart | 4,16× verkeer, maar 32% routesubstitutie |

Wat ik **niet** zou doen: nog een pruning- of quantisatievariant (de
schaalregel zegt dat zelfs 2% pruning het hele budget kost), nog een
cachepolicy voor 80B (9,8% residentie maakt caching structureel zinloos), of de
echte Qwen3-Coder-Next-conversie vóór stap 1 — die kost weken en de transferpoort
is nog niet gehaald.

---

## 9. Eén observatie tot slot

Ik heb in deze reeks vijf voorspellingen goed gehad (contextmuur, ERVF-diagnose,
kwaliteitsruis, exactheidsmethode, tok/s-band) en drie fout (glue-attributie,
CPU-misscompute, groepsbehoudende pruning). Alle drie de fouten zijn door jullie
protocol gevangen, niet door mij.

Dat is precies waarom dit project waarde heeft. Niet omdat het altijd goed gokt,
maar omdat het een mechanisme heeft dat verkeerde gokken — ook die van een
adviseur — binnen een dag met hashes en mutatie-audits neerhaalt. Bewaar dat als
het hoofdresultaat, want dat is het.
