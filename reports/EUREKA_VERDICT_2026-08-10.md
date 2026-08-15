# DeepSeek-V2-Lite — Eureka-verdict over behavioral compression

Datum: 2026-08-10  
Model: `deepseek-ai/DeepSeek-V2-Lite` Base  
Modelcommit: `604d5664dddd88a0433dbae533b7fe9472482de0`  
WikiText-commit: `b08601e04326c79dfdd32d625aee71d232d685c3`

## Uitkomst in één zin

Er is overtuigend aangetoond dat DeepSeek-V2-Lite veel functioneel equivalente
routes heeft en dat een zeer conservatieve cache-aware routevervanging
modelbreed 11–12,5% expert-loads bespaart zonder meetbare CE-verslechtering op
de geteste corpora; de praktische regel is echter al vrijwel gepubliceerd als
`Max Rank`, zodat dit een geslaagde onafhankelijke reproductie en diagnostische
Eureka is, geen nieuwe fundamentele uitvinding.

## Drie verschillende oordelen

| Vraag | Oordeel | Bewijsniveau |
|---|---|---|
| Bestaan er veel gedragsmatig equivalente routes? | **Ja** | uitputtende 924-route-interventie, held-out test |
| Kan een simpele inzetbare cachepolicy daar veilig iets van benutten? | **Ja, bescheiden** | alle 26 MoE-lagen, vier corpussplits, bootstrap, korte KV-rollout |
| Is die praktische policy nieuw? | **Nee** | primaire prior art bevat vrijwel dezelfde Max-Rankregel en sterkere systemen |

“Eureka” betekent in dit rapport daarom: een echte en reproduceerbare
modelbevinding met praktisch signaal. Het betekent niet: patentwaardige
nieuwheid, production-ready runtime of oplossing van de oorspronkelijke
10-tokens/s-K3-moonshot.

## Vooraf gestelde hypotheses en gates

| Hypothese | Gate | Resultaat | Verdict |
|---|---:|---:|---|
| Behavioral observability is laag-rank | rank 64 bevat 80–90% sensitivity | 33,14%; `r80=609`, `r90=943` | **faalt** |
| Projected oracle herstelt 3-bitmodel | ≥70% KL/CE-schade bij rank 64 | test: 13,30% KL, 18,60% CE | **faalt** |
| 3→4-bit precisie is schaars | ≤25% perfecte upgrades evenaart 4-bit | test: 23,812%; bij 25% KL 0,003193 versus 0,003245 | **oracle slaagt** |
| Een teacher-free precisionselector vindt die events | benadert oracle op test | beste voorspellers blijven duidelijk slechter dan all-4-bit | **faalt** |
| 2→4-bit precisie is schaars | ≤25% upgrades | 42,546%, gemiddeld 2,851 actieve bits | **faalt** |
| Routes vormen niet-triviale equivalentieklassen | meerdere lage-KL alternatieven | 85,55% heeft alternatief bij KL ≤0,001 | **slaagt** |
| Equivalentie overleeft downstreamlagen | lage eind-KL na eerdere interventie | laag 23-regel: KL 0,001396, top-1 99,22% | **slaagt** |
| Modelbrede praktische cachepolicy | minder loads, CE binnen ±2% | 11–12,5% minder loads; CE circa nul | **slaagt** |
| Nieuwe fundamentele methode | geen directe eerdere methode | Max Rank, MoE-ERAS, BuddyMoE e.a. bestaan | **faalt** |
| Production deployment | packed kernel + lange rollouts + latency | niet uitgevoerd | **niet bewezen** |

## 1. Negatief bewijs: de observability-patch is dood

De centrale QuotientQuant-hypothese voorspelde dat alleen een kleine subruimte
van hidden-statefouten zichtbaar zou zijn in toekomstige logits. Met acht
onafhankelijke Fisher-samples per trainstate op laag 26 blijkt het spectrum veel
te breed:

- effective/participation rank: `260,21`;
- `r80=609`, `r90=943`, `r95=1228`;
- rank 64 bevat slechts `33,14%` van de behavioral trace;
- rank 64 herstelt op test `13,30%` van 3-bit-KL-schade en `18,60%` van
  CE-schade;
- zelfs rank 512 herstelt slechts `33,90%` van de KL-schade.

Een damage-weighted error-PCA lijkt lokaal compacter, maar voorspelt de held-out
modelschade niet beter. Een echte projected weight patch is daarom terecht niet
gebouwd: het oracleplafond mist de gate al ruim.

Bron: `baseline/layer26_behavioral_observability_reliability.json`.

## 2. Gedeeltelijk positief bewijs: dynamic precision heeft alleen een oracle

Voor elk token zijn alle 64 mogelijke subsets van de zes actieve experts exact
als 3- of 4-bit uitgevoerd. Op test:

- all-3-bit: KL `0,020321`, relatieve CE `+1,212%`;
- all-4-bit: KL `0,003245`, relatieve CE `+0,212%`;
- perfect 3→4-oracle: `23,812%` expertupgrades om all-4-bit-KL binnen 1% te
  evenaren;
- 25% upgrades: gemiddelde actieve precisie 3,25 bit, KL `0,003193`, relatieve
  CE `+0,162%`.

Dat is een echte rate-distortionopening. Zij wordt niet inzetbaar: router-,
delta-, quadratic-mask- en progressive-bitplanevoorspellers generaliseren niet
naar het oracle. Het agressievere 2→4-oracle vereist `42,546%` upgrades en mist
de gate zelf al.

Bronnen:

- `baseline/layer26_dynamic_precision_exact_oracle.json`;
- `baseline/layer26_dynamic_precision_predictors.json`;
- `baseline/layer26_quadratic_mask_predictor.json`;
- `baseline/layer26_progressive_bitplane_predictor.json`;
- `baseline/layer26_dynamic_2to4_exact_oracle.json`.

## 3. Positief bewijs: routes zijn functioneel niet uniek

Voor ieder van 256 validation- en 256 testtokens zijn in laag 26 alle
`C(12,6)=924` routes uit de top-12 exact geëvalueerd. De originele routerweights
zijn behouden; DeepSeek hernormaliseert top-6 niet en deze test doet dat dus ook
niet. Er is geen coefficientfit of teachergetrainde selector.

### Held-out test

| KL-grens | Tokens met ≥1 alternatief | Gem. alternatieven | Gem. entropy incl. origineel |
|---:|---:|---:|---:|
| 0,0001 | 43,75% | 82,29 | 2,477 bits |
| 0,001 | **85,55%** | **202,00** | **5,087 bits** |
| 0,003 | 99,22% | 348,98 | 7,305 bits |
| 0,01 | 100% | 517,92 | 8,539 bits |

De beste niet-originele route heeft gemiddelde KL `0,000427` en p95
`0,001786`. Wanneer maximaal de helft van de experts overlapt
(`Jaccard ≤ 0,5`), is de beste route nog steeds gemiddeld `0,000974` KL met p95
`0,002826`. Een volledig disjuncte route is niet equivalent: gemiddelde KL
`0,141261`. Het effect is dus groot én gestructureerd.

Bron: `baseline/layer26_route_equivalence.json`.

### Downstreaminterventie

Een route-interventie in laag 23 is gevolgd door de exacte lagen 24–26. De
teacher-free vaste regel “vervang routerrangen 5–6 door 7–8” geeft op test:

- eind-KL `0,001396`;
- top-1-overeenkomst `99,22%`;
- relatieve CE-delta `+0,112%`;
- downstream top-6-overlap blijft ongeveer `98,1–98,7%`.

Hiermee is de bevinding niet alleen een lokale outputcoïncidentie.

Bron: `baseline/layer23_route_equivalence_downstream.json`.

## 4. Praktische modelbrede policy

De conservatieve policy per token en laag:

1. behoud routerrangen 1–5;
2. vergelijk de exacte within-token LRU-misses van top-6 met
   top-5 + routerrang 7;
3. gebruik rang 7 uitsluitend wanneer die route onmiddellijk minder misses
   veroorzaakt;
4. behoud de originele, niet-hergenormaliseerde routerweight van de gekozen
   expert;
5. vergelijk aggregate loads met de ongewijzigde teacherroutes in een
   onafhankelijke strict LRU-cache.

Capaciteit is 32 experts **per MoE-laag**. Elke 128-tokensequentie start met een
lege cache. Kwaliteit is exact in BF16 uitgevoerd over alle 27 lagen.

### Hoofdresultaten

| Corpus/split | Tokens | Loadreductie | KL | Top-1 | Relatieve CE-delta | 95%-CI CE |
|---|---:|---:|---:|---:|---:|---:|
| WikiText validation | 1.024 | onderdeel totaal | 0,004303 | 96,19% | −0,018% | [−0,363%, +0,357%] |
| WikiText test | 1.024 | 10,09–11,28% CI | 0,003861 | 97,36% | +0,097% | [−0,130%, +0,328%] |
| Nederlandse research/instructies | 1.024 | 10,75–12,40% CI | 0,007199 | 95,51% | −0,006% | [−0,214%, +0,199%] |
| Pythoncode | 1.024 | 12,16–14,86% CI | 0,002614 | 98,14% | −0,082% | [−0,325%, +0,213%] |

Aggregate per corpus:

- WikiText: `92.607 → 82.221` loads, **11,215% reductie**;
- instructies + code: `95.762 → 83.797` loads, **12,495% reductie**.

Alle 32 individuele 128-tokenblokken hebben positieve loadreductie. De
bootstrap gebruikt 10.000 resamples met sequentieblokken als sampling units.
Alle vier CE-intervallen bevatten nul; dit bewijst geen exacte gelijkheid, maar
wel dat op deze steekproef geen betrouwbare CE-schade zichtbaar is.

Bronnen:

- `baseline/modelwide_cache_aware_bottom1_teacher_lru_capacity32_wikitext_1024_ci.json`;
- `baseline/modelwide_cache_aware_bottom1_teacher_lru_capacity32_diverse_1024_ci.json`.

## 5. Autoregressieve KV-cache-smoke

Prompt: `Explain in one paragraph why mixture-of-experts routing can be redundant.`

- onafhankelijke greedy teacher/studentprefixes;
- echte `DynamicCache`, eindlengte 19;
- 4/4 gegenereerde tokens identiek: `\n\n## `;
- pre-decision teacher→student-KL per stap:
  `0,00795`, `0,00815`, `0,01408`, `0,01872`;
- totale loads: `1.343 → 1.283`, oftewel `−4,47%`;
- minimum routeroverlap per stap: `5/6`.

Door divergerende cachehistorie kunnen losse stappen meer adaptive dan strict
loads hebben; alleen het totale traject is positief. Vier tokens is een smoke,
geen bewijs voor de vooraf gewenste 512 tokens. De referentie-implementatie
kost circa 63 seconden per token en meet geen deploymentefficiëntie.

Bron: `baseline/cache_aware_teacher_lru_kv_rollout_4tokens.json`.

## 6. Geheugen- en I/O-accounting

Eén routed expert bevat `8.650.752` parameters:

- BF16: `17.301.504` bytes = 16,50 MiB;
- hypothetische packed int4 plus BF16-rijschalen: `4.335.104` bytes =
  4,134 MiB.

DeepSeek-V2-Lite heeft 26 × 64 routed experts:

- routed BF16-bank: `26,8125 GiB`;
- routed int4-bank: `6,7182 GiB`;
- overige checkpointweights in BF16: `2,4431 GiB`;
- overige BF16 + hele routed int4-bank: `9,1613 GiB`.

Een cache van 32 experts per elk van 26 lagen vraagt:

- BF16: `13,4063 GiB`;
- int4: `3,3591 GiB`;
- overige BF16 + int4-cache: `5,8022 GiB`.

De gemeten loadcounts projecteren bij int4 naar:

| Corpus | Strict routed I/O/token | Adaptive | Bespaard | Bespaarde bandbreedte bij 10 tok/s |
|---|---:|---:|---:|---:|
| WikiText | 186,94 MiB | 165,98 MiB | 20,97 MiB | 0,205 GiB/s |
| Instructies + code | 193,31 MiB | 169,16 MiB | 24,15 MiB | 0,236 GiB/s |

Dit zijn deterministische byteprojecties, geen fysieke transfers. De run heeft
geen packed-int4gewichten, async prefetch, unified cross-layer cache of custom
kernel uitgevoerd. Aandacht, shared experts, compute, metadata en KV-verkeer
zijn niet in deze bandbreedte opgenomen. Routepolicy plus int4 moet afzonderlijk
op kwaliteit worden gevalideerd.

Bron: `baseline/route_cache_storage_io_accounting.json`.

## 7. Waarom dit geen nieuwe uitvinding is

De primaire novelty-audit vindt directe overlap:

- **Mixture of Cache-Conditional Experts** (TMLR 2025) introduceert trainingvrije
  cache-aware routing, test DeepSeek-V2-Lite en bevat `Max Rank`; onze regel is
  vrijwel het grensgeval `J=5, M=7`:
  <https://arxiv.org/abs/2412.00099>.
- **MoE-ERAS** kiest experts al op kwaliteit én residentie:
  <https://openreview.net/forum?id=o43eHjPEMO>.
- **BuddyMoE** profileert vervangingsparen en gebruikt resident buddies bij
  misses: <https://arxiv.org/abs/2511.10054>.
- **SERE** reroutet secundaire experts op activatiesimilariteit en rapporteert
  DeepSeekV2-speedups: <https://arxiv.org/abs/2602.07616>.
- **Counterfactual Routing Analysis** vergelijkt al sampled equal-compute routes
  en bevat DeepSeek-V2-Lite: <https://arxiv.org/abs/2605.07260>.
- **SliceMoE** combineert dynamische bit-slices, caching en DeepSeek-V2-Lite:
  <https://arxiv.org/abs/2512.12990>.

Onze uitputtende `top12 choose 6`-telling onder volledige-vocabulaire-KL lijkt
in deze gerichte zoekronde een scherpere diagnostiek dan het sampled
counterfactual protocol. Dat rechtvaardigt hoogstens een mogelijke
meetinstrumentbijdrage, niet de claim dat routevervanging of cache-aware routing
nieuw is. Er is geen patentonderzoek uitgevoerd.

Volledige vergelijking: `../docs/PRIOR_ART.md`.

## 8. Definitief stop/go-besluit

### Stop

- Geen projected observability-patches bouwen.
- Geen verdere eenvoudige teacher-free 2/3/4-bitpredictors trainen zonder een
  nieuw signaal; de huidige families zijn door held-out tests weerlegd.
- De bottom-rank cachepolicy niet als eigen uitvinding presenteren.
- Nog niet naar V4 Flash opschalen op basis van deze cijfers.

### Go

- Behoud route-equivalence entropy als reproduceerbare diagnostiek.
- Gebruik de modelbrede policy als gecontroleerde V2-Lite-baseline tegen
  gepubliceerde Cache-Prior, BuddyMoE en SliceMoE.
- De volgende zinvolle engineeringproef is een echte packed runtime met een
  unified byte-budgetcache, langere GSM8K/code-rollouts en wall-clock/energie.
  Pas zo'n implementatie kan aantonen of de gemeten 11–12,5% loads ook latency
  oplevert.

Een 512-tokenrun met de huidige 63 s/token-referentie zou ongeveer negen uur
kosten en uitsluitend een reeds gepubliceerde policy verder valideren; zonder
packed runtime is dat geen rationele volgende bewijsstap.

## 9. Reproductie

Belangrijkste commando's vanuit de projectroot:

```powershell
.\.venv\Scripts\python.exe scripts\estimate_layer26_observability.py
.\.venv\Scripts\python.exe scripts\evaluate_layer26_dynamic_precision_oracle.py
.\.venv\Scripts\python.exe scripts\evaluate_layer26_route_equivalence.py
.\.venv\Scripts\python.exe scripts\evaluate_layer23_route_equivalence_downstream.py
.\.venv\Scripts\python.exe scripts\evaluate_modelwide_cache_aware_bottom1.py --blocks-per-split 8 --capacity 32 --corpus-preset wikitext --report-name modelwide_cache_aware_bottom1_teacher_lru_capacity32_wikitext_1024_ci.json
.\.venv\Scripts\python.exe scripts\evaluate_modelwide_cache_aware_bottom1.py --blocks-per-split 8 --capacity 32 --corpus-preset local_diverse --report-name modelwide_cache_aware_bottom1_teacher_lru_capacity32_diverse_1024_ci.json
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\build_route_cache_accounting.py
.\.venv\Scripts\python.exe -m pytest -q
```

De oudere `RESULTS_2026-08-09.md` blijft het verslag van de eerste
compressiebaseline. Dit document is het actuele eindverdict voor QuotientQuant,
dynamic precision en route-equivalentie.
