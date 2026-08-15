# Offload-roofline analyse — waarom CRAFT/HERA de verkeerde term optimaliseert

**Datum:** 2026-08-11 · **Status:** externe analyse, geen gemeten resultaat · **Type:** afleiding + preregistratievoorstel

Alle getallen hieronder komen uit (a) de eigen HERA/E2GQ-artifacts in deze bundel,
(b) de in de research geciteerde WASTE/K3-cijfers, (c) elementaire bandbreedte-algebra.
Niets hiervan is zelf gemeten. Elk cijfer is bedoeld om gefalsificeerd te worden.

---

## 0. Reproductiecontrole

De eigen cijfers zijn eerst exact nagerekend, zodat de rest van de afleiding op
dezelfde basis staat:

| Grootheid | Nagerekend | Gerapporteerd |
|---|---:|---:|
| E2GQ code-entropie | 1.782864891374 bpp | 1.782864891374 bpp |
| + BF16 group-128 scales | 1.907864891374 bpp | 1.907864891374 bpp |
| Qwen3 expertgrootte (3·2048·768) | 4 718 592 par = 9.000 MiB BF16 | 9.000 MiB |
| HERA hot-unie 6081 @1.930709 bpp | 6.449 GiB | 6.449 GiB |
| + INT4 trunk | 7.167 GiB | 7.167 GiB |
| general→code nieuwe experts (uit Jaccard) | 1229 | 1229 |

Alles sluit. De afleiding hieronder gebruikt dezelfde constanten.

---

## 1. De harde ondergrens: de trunk, niet de experts

Decodeersnelheid bij batch 1 is een roofline met **twee** termen:

```
bytes/token  =  T / s   +   E · (1 − h) · c
                ▲            ▲
                trunk        experts
   T = per token volledig actieve niet-expert gewichten
   s = speculatiediepte (geaccepteerde tokens per verificatiepass)
   E = expertbytes/token ongecached, h = cache-hitrate, c = compressiefactor
```

Voor K3 met de in de research geciteerde cijfers: **T = 27.28 GB**, E ≈ 17 GB @3 bpp.

De trunk is *elke* token volledig actief. Dus:

| Doel | Trunk vraagt alleen al |
|---:|---:|
| 1 tok/s | 27.3 GB/s |
| 5 tok/s | 136.4 GB/s |
| **10 tok/s** | **272.8 GB/s** |

Wat de bussen leveren:

| Bus | GB/s | Max tok/s op trunk alleen |
|---|---:|---:|
| PCIe 4.0 x8 | 14 | **0.51** |
| PCIe 4.0 x16 | 25 | 0.92 |
| PCIe 5.0 x16 | 50 | 1.83 |
| DDR5 dual channel | ~60 | 2.20 |
| RTX 5090 VRAM | 1790 | 65.6 |

**Gevolg.** Op 8 GiB VRAM past de 27.28 GB trunk niet, dus hij moet per token over
PCIe. Het plafond is dan **0.5–0.75 tok/s**, ongeacht hoe goed de experts worden
gecomprimeerd. WASTE meet 0.29–0.62 tok/s: die implementatie zit dus **al op de
fysieke grens**, en die grens wordt door de trunk gezet.

> **K3 op ≥10 tok/s met 8 GiB VRAM is geen open onderzoeksvraag. Het is
> rekenkundig gesloten.** De enige oplossing is een trunk-residente VRAM van
> ≥ 27.28 GB + KV + workspace ≈ 32 GB.

Alle negen geteste hypotheses (H1–H8, RSIV, FLEQ, E2GQ, HERA) vallen uitsluitend
op de **tweede** term. Die is bij K3 op deze hardware ≈ 38% van het verkeer, en
de eerste term is niet comprimeerbaar met expertmethodes.

**Falsificatie van deze sectie:** meet de werkelijke per-token actieve
niet-expertbytes van K3 en de gerealiseerde H2D-bandbreedte. Als T ≪ 27.28 GB
of als de bus > 100 GB/s haalt, vervalt de conclusie.

---

## 2. De drie hefbomen, op schaal

| Hefboom | Winst | Kosten | Werkt op |
|---|---:|---|---|
| Expertcompressie 3→2 bpp | ×1.47 | ~8 dagen + kwaliteitsrisico | term 2 |
| 50% atom-sparsity (H3, gemeten CE +0.04%) | ×1.72 | al bewezen | term 2 |
| Cache-conditional routing (Mass-Budget) | ×1.16 | al bewezen | term 2 |
| **alle CRAFT-software samen** | **×2.93** | | term 2 |
| RAM 32 → 128 GiB (hit 30%→80%) | ×3.50 | ~€300, nul onderzoek | term 2 |
| Speculative decoding, 8 geaccepteerd | ×8.0 | bekend, prior art | **term 1** |

Eén RAM-upgrade verslaat het volledige softwareprogramma. Speculative decoding is
de enige hefboom die de dominante term aanpakt.

---

## 3. Expert-populariteitscurve

Gefit op drie exacte ankers uit de eigen E2GQ-run (6144 paren, 12 582 912
invocaties, 1695 paren met count<128 die samen 0.381136% van de calls dragen):

| resident N | % bank | dekking (lognormaal) | miss-calls/token (van 384) |
|---:|---:|---:|---:|
| 1536 | 25.0% | 87.6% | 47.7 |
| 2428 | 39.5% | 94.1% | 22.7 |
| 3072 | 50.0% | 96.6% | 12.9 |
| 4449 | 72.4% | 99.24% | 2.9 |

Het exacte meetanker C(4449) = 99.619% ligt boven beide fits, dus het model is
**conservatief** — echte hitrates liggen hoger dan hier geprojecteerd.

---

## 4. Wat hieruit volgt voor Qwen3-30B-A3B (huidige laptop)

8 GiB VRAM − KV@8k (0.750 GiB) − CUDA/workspace (1.0) − INT4 trunk (0.718)
= **5.532 GiB expertcache**.

| Expertformaat | Slots | Hostbank | Miss/token | PCIe-tijd | Projectie |
|---|---:|---:|---:|---:|---:|
| **4-bit GPTQ** | 2441 (39.7%) | 13.92 GiB | 22.4 calls = 54.5 MB | 3.9 ms | ~40–100 tok/s |
| 3-bit | 3222 (52.4%) | 10.55 GiB | 11.3 calls = 20.8 MB | 1.5 ms | ~50–130 tok/s |
| 2-bit entropy | 5216 (84.9%) | 6.52 GiB | 0.8 calls = 0.9 MB | 0.1 ms | ~50–160 tok/s |

**De hele 2-bit/entropy-inspanning lost een probleem op dat de cache al
oplost.** Bij gewone 4-bit GPTQ is het misverkeer 54 MB/token; dat is 4 ms op een
Gen4 x8 bus. De FLEQ-mislukking, het E2GQ-coverageprobleem en de HERA-geheugengate
verdwijnen alle drie zodra de volledige bank in host-RAM staat (13.92 GiB van 32).

---

## 5. HERA is met de verkeerde vraag gesloten

`static_tier_negative` meet de **unie** over vijf domeinen: 6081/6144 → 7.167 GiB
tegen de 5.75-gate (+24.6%). Maar de per-domein hotsets zijn veel kleiner:

| Domein | Hot | Resident (GiB) | 5.75-gate |
|---|---:|---:|---|
| code | 4168 | 5.138 | **PASS** (−10.6%) |
| multilingual | 4320 | 5.299 | **PASS** (−7.8%) |
| general | 4453 | 5.440 | **PASS** (−5.4%) |
| math | 4823 | 5.833 | FAIL (+1.4%) |
| instruction | 4957 | 5.975 | FAIL (+3.9%) |
| *statische unie* | *6081* | *7.167* | *FAIL (+24.6%)* |

Kosten van een domeinwissel, afgeleid uit de gemeten Jaccard-matrix:

| Wissel | Nieuwe experts | Bytes | Tijd @15 GB/s |
|---|---:|---:|---:|
| general → code | 1229 | 1.303 GiB | 89 ms |
| code → instruction | 1615 | 1.713 GiB | 117 ms |
| worst case | 1615 | 1.713 GiB | **117 ms, eenmalig** |

Een eenmalige 117 ms bij een domeinwissel, geamortiseerd over honderden tokens,
tegenover een gate-overschrijding van 24.6%. De conclusie `static_tier_negative`
is correct voor statische tiers en zegt niets over online residency.

---

## 6. Waarom tile-64 faalde (en de enige mogelijke fix)

Bij willekeurige neuronvolgorde en selectiefractie f = 0.25 geldt voor een tile
van 64 neuronen:

```
P(tile bevat geen enkel geselecteerd neuron) = (1 − 0.25)^64 = 1.0 × 10⁻⁸
```

Dus **elke** tile wordt aangeraakt en tile-granulariteit levert bij gelijke
kwaliteit nul bytebesparing. Bij gelijk *bytebudget* moet tile-64 hele blokken
kiezen en vangt het de verspreide nuttige neuronen niet — dat verklaart de
gemeten 8.5× KL-kloof volledig, zonder dat er iets mis is met het idee.

De enige fix is een **offline permutatie**: bouw de co-selectiegraaf over de
intermediate neuronen uit de reeds opgeslagen masks, partitioneer met
spectral clustering of METIS in blokken van 64, permuteer gate/up-rijen en
down-kolommen één keer. Nul runtimekosten, volledig verifieerbaar
(gepermuteerde reconstructie moet bit-identiek zijn).

---

## 7. Voorgestelde preregistraties, op volgorde van zekerheid

### P-A · Eerste wall-clock (P(pass) ≈ 0.95)

**Hypothese.** Qwen3-30B-A3B draait op deze laptop op ≥10 tok/s batch-1 decode,
relatieve CE ≤2%, VRAM ≤8 GiB, proces-RSS ≤32 GiB.

**Constructie.** 4-bit GPTQ over de hele bank (13.92 GiB) in gepind host-RAM;
INT4 trunk resident in VRAM; LFU-expertcache van 2400 slots in VRAM; async
prefetch zodra routerlogits van laag ℓ bekend zijn.

**Gates.** tok/s ≥10 · CE ≤2% · VRAM ≤8 GiB · 512-token rollouts stabiel ·
gemeten miss-calls/token gerapporteerd naast de geprojecteerde 22.4.

**Waarom dit eerst moet.** Er bestaat na negen hypotheses geen enkel tok/s-getal.
Zonder dat getal is geen enkele claim in dit project verifieerbaar, en met dit
getal wordt elke volgende claim een gemeten delta.

### P-B · Online residency verslaat statische tiering (P ≈ 0.90)

**Hypothese.** Bij een vast VRAM-budget van 4700 expertslots houdt een
LFU-cache met router-lookahead-prefetch de gemiddelde cold-calls/token onder 3.0
op alle vijf domeinen, en herstelt hij na een domeinwissel binnen 200 tokens.

**Meting.** Puur op de bestaande route-traces, geen GPU nodig. Rapporteer de
volledige misscurve C(N) voor N ∈ {1024 … 6144} per domein, plus de
hersteltrajecten voor alle 20 gerichte domeinwissels.

**Gates.** mean cold-calls/token ≤3.0 op alle domeinen · p99 ≤12 ·
hersteltijd ≤200 tokens · resident ≤5.75 GiB.

### P-C · Roofline-falsificatie van het K3-doel (P ≈ 0.85)

**Hypothese.** tok/s ≤ min(BW_trunk / T, BW_expert / E_eff), en op 8 GiB VRAM
geldt voor K3 een plafond ≤1.0 tok/s.

**Meting.** Directe microbenchmark: gepinde H2D-doorvoer, werkelijke per-token
actieve niet-expertbytes, en een 64-token decode van een model waarvan de trunk
bewust niet in VRAM past. Voorspelling vooraf vastleggen, dan meten.

**Waarom dit waardevol is.** Dit is de enige uitspraak in het hele project die
alle negen negatieve resultaten in één model verklaart, en hij is publiceerbaar
als roofline voor offloaded MoE-inferentie.

### P-D · Speculative decoding als trunk-amortiseerder (P ≈ 0.70)

**Hypothese.** Bij draftdiepte 8 wordt de trunkterm gedeeld door het aantal
geaccepteerde tokens, terwijl de expertunie sublineair groeit:
`E[uniek] = E_layer · (1 − (1 − k/E_layer)^s)`.

Voor K3 (896 experts, top-16, s=8): 118.6 unieke van 128 → slechts ×1.08 op de
expertterm. **De winst zit volledig in term 1.**

**Meting.** Acceptatiegraad van (a) het gefalsificeerde H3 25%-atoommodel als
drafter en (b) een klein extern draftmodel; plus de gemeten U(s)/(k·s)-curve.

**Gate.** ≥4.0 geaccepteerde tokens per verificatiepass bij CE-neutrale output.

### P-E · Neuronpermutatie redt tile-granulariteit (P ≈ 0.75)

**Hypothese.** Na offline co-activatiepartitionering haalt tile-64 bij 25%
atoombudget ≤1.20× de neuron-oracle-KL (huidig: 8.5×).

**Controle.** Gepermuteerde volledige reconstructie moet bit-identiek zijn aan
de originele expertoutput.

---

## 8. Wat ik zou stoppen

- **Nog een expert-compressievariant.** De volledige softwareladder is ×2.93 en
  de hardware-hefbomen zijn ×3.5 en ×8.0.
- **2-bit en entropy coding op de huidige testbed.** Sectie 4 laat zien dat de
  cache dat probleem al oplost; 4-bit GPTQ is voldoende en risicoloos.
- **Nieuwheidsgates als stopcriterium.** 19.65% unie-reductie en 14%
  loadreductie zijn geen mislukkingen; het zijn winsten die zijn weggegooid omdat
  ze geen paper rechtvaardigden. Systeemgates (tok/s, VRAM, CE) horen te
  beslissen; nieuwheid is een aparte, latere vraag.
