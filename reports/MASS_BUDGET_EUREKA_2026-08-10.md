# DeepSeek-V2-Lite — Mass-Budget Cache-Prior-verdict

Datum: 2026-08-10  
Model: `deepseek-ai/DeepSeek-V2-Lite` Base  
Modelcommit: `604d5664dddd88a0433dbae533b7fe9472482de0`  
WikiText-commit: `b08601e04326c79dfdd32d625aee71d232d685c3`

## Uitkomst in één zin

Er is een **begrensde praktische Eureka** bereikt: de nieuwe trainingvrije
`Mass-Budget Cache-Prior`-policy verslaat de oude rank-7-policy overtuigend en
verbetert op meerdere modelbrede evaluaties het kwaliteit-versus-loadfront van
een vaste Cache-Prior; een vooraf vastgelegde bevestiging op 4.096 nog niet
bekeken WikiText-tokens haalt alle vijf succesgates. Niet bewezen zijn
universele autoregressieve dominantie, fysieke versnelling en fundamentele
nieuwheid.

## Wat precies wel en niet bewezen is

| Vraag | Oordeel | Sterkste bewijs |
|---|---|---|
| Is de evaluator numeriek betrouwbaar? | **Ja** | `original`: KL/CE/fout exact `0` in alle modelbrede runs |
| Reproduceren we de primaire Cache-Prior-baseline? | **Ja** | DeepSeek-validatiemiss `26,29%→6,53%` bij λ=0,5 |
| Is Mass-Budget sterker dan onze oude rank-7-policy? | **Ja** | vooraf vastgelegd testvenster: meer loadreductie én lagere KL |
| Verbetert Mass-Budget het vaste-λ-Paretofront? | **Ja, binnen deze evaluaties** | discrete dominantie op exploratieve WikiText- en domeintransferruns; vooraf bevestigde ~15% lagere KL bij een maximaal 2 pp loadmarge |
| Heeft δ=0,004 een aantoonbare CE-regressie? | **Nee gevonden** | prereg test `−0,057%`, 95%-CI `−0,171%–+0,060%` |
| Geeft de routepolicy een formele outputrisicogarantie? | **Nee** | de conformal slate-dekking generaliseert onvoldoende |
| Domineert zij in autoregressieve generatie? | **Nee** | één korte rollout is een expliciet tegenvoorbeeld |
| Is wall-clockwinst bewezen? | **Nee** | alleen expert-loads en byte-accounting, geen packed kernel |
| Is de methode fundamenteel nieuw? | **Niet vastgesteld** | gerichte zoekactie vindt geen exact equivalent, maar zeer nabije prior art |

## Methode

Voor een token is `R₀` de originele top-6-route en `p(e)` de ongewijzigde
softmax-routerkans. We houden de top-2 altijd vast. Vervolgens genereren we een
vaste, trainingvrije slate van Cache-Prior-routes met

```text
λ ∈ {0,025; 0,05; 0,075; 0,10; 0,125; 0,15; 0,20; 0,25;
     0,30; 0,40; 0,50; 0,75; 1,00}.
```

Voor iedere kandidaatroute `R` berekenen we uitsluitend met routerinformatie:

```text
Δmass(R) = Σ[e∈R₀] p(e) − Σ[e∈R] p(e).
```

`Mass-Budget(δ)` kiest uit de routes met `Δmass(R) ≤ δ` de route met de minste
onmiddellijke LRU-misses. Bij gelijkstand wint eerst de kleinste mass loss en
daarna de kleinste λ. De originele routerkansen — niet de gebiaste logits —
blijven de geselecteerde expertoutputs wegen.

De policy gebruikt bij runtime dus geen teacherlogits, expertoutputs, getraind
risicomodel of labels. Zij gebruikt alleen de huidige routerlogits, de
cachetoestand, een per laag op validatietokens geschatte gemiddelde logitrange
en δ. De huidige Pythonreferentie sorteert wel dertien keer per token; een
geoptimaliseerde implementatie en latencytest ontbreken nog.

## 1. Integriteit en primaire reproductie

Iedere modelbrede run injecteert alleen de gekozen gate-ID's in de volledige
officiële DeepSeek-decoderlaag en gebruikt de officiële MoE-kernel. De
`original`-policy reproduceert alle 26 lagen met maximale absolute fout `0` en
geeft aan het eind KL `0`, CE-delta `0`, top-1 `1` en loadreductie `0`.

De paper-faithful 1.024-tokenrun gebruikt contextlengte 1.024, top-2-bescherming,
capaciteit 32 per MoE-laag, originele routergewichten en de beschreven
within-token LRU-volgorde.

| Split/policy | miss | loadreductie | KL | relatieve CE | top-1 |
|---|---:|---:|---:|---:|---:|
| validatie original | 26,29% | 0% | 0 | 0% | 100% |
| validatie λ=0,5 | 6,53% | 75,14% | 0,028777 | +0,202% | 94,34% |
| test original | 31,50% | 0% | 0 | 0% | 100% |
| test λ=0,1 | 17,47% | 44,54% | 0,006074 | +0,072% | 97,46% |
| test λ=0,5 | 8,48% | 73,06% | 0,026684 | +1,265% | 93,75% |

Dit ligt zeer dicht bij het gepubliceerde DeepSeek-signaal van ongeveer
`28%→7%` miss en `0,1–3%` perplexiteitstoename; onze test-PPL-ratio bij λ=0,5
is `+1,943%`. Daarmee is de sterke vaste Cache-Prior een echte baseline en
geen stroman.

Bron: `baseline/paper_context1024_wikitext_cache_prior_aggressive.json`.

## 2. Vooraf vastgelegde bevestiging

Na de exploratieve ontwikkeling zijn vóór inspectie van het nieuwe venster
policy's, offset en vijf gates vastgelegd in
`PREREGISTERED_MASS_BUDGET_CONFIRMATION_2026-08-10.md`. Daarna zijn vanaf
tokenoffset 4.096 onafhankelijk 16×128 validatie- en 16×128 testtokens
geëvalueerd. Er is niets herafgesteld.

### Testresultaten

| Policy | loadreductie (95%-CI) | KL (95%-CI) | relatieve CE (95%-CI) | top-1 |
|---|---:|---:|---:|---:|
| oude Max-Rank J5/M7 | 10,559% (10,178–10,958) | 0,004389 (0,004087–0,004706) | −0,007% (−0,136–+0,132) | 96,34% |
| vaste λ=0,0275 | 15,410% (14,624–16,272) | 0,004363 (0,003992–0,004762) | +0,053% (−0,050–+0,157) | 96,00% |
| **Mass-Budget δ=0,004** | **14,017% (13,253–14,860)** | **0,003704 (0,003421–0,004017)** | **−0,057% (−0,171–+0,060)** | **96,58%** |
| vaste λ=0,095 | 42,195% (40,932–43,550) | 0,011255 (0,010426–0,012065) | +0,084% (−0,214–+0,382) | 94,09% |
| Mass-Budget δ=0,018 | 40,271% (38,955–41,653) | 0,010137 (0,009398–0,010871) | +0,074% (−0,120–+0,248) | 94,68% |

De vooraf gekoppelde lage policy haalt op validatie én test circa 15% lagere
KL dan λ=0,0275, terwijl de loadreductie respectievelijk 1,379 en 1,393
procentpunt lager ligt — binnen de vooraf vastgelegde marge van 2 punten.
Tegen de oude rank-7-regel is δ=0,004 op test tegelijk beter in load en KL.
Alle zestien testblokken besparen loads. Alle vijf gates slagen.

Bron: `baseline/preregistered_wikitext_offset4096_mass_budget_confirmation.json`.
De onafhankelijk herberekende gate-uitkomst staat in
`baseline/preregistered_mass_budget_confirmation_gates.json`.

## 3. Exploratieve front- en transfertests

De onderstaande punten zijn waardevol voor transfer, maar zijn niet zo sterk
als de preregistratie omdat de methode al bekend was toen ze werden bekeken.

| Corpus/context | vergelijking op test | loadreductie | KL | interpretatie |
|---|---|---:|---:|---|
| WikiText 8×128 | λ=0,025 | 11,90% | 0,003203 | — |
| WikiText 8×128 | **δ=0,004** | **12,61%** | **0,003018** | discrete dominantie |
| WikiText 8×128 | λ=0,085 | 35,64% | 0,007611 | — |
| WikiText 8×128 | **δ=0,016** | 35,40% | **0,006714** | 0,23 pp minder load, 11,8% lagere KL |
| instructie→code 8×128 | λ=0,0275 | 21,46% | 0,002617 | — |
| instructie→code 8×128 | **δ=0,004** | **22,24%** | **0,002404** | discrete dominantie |
| instructie→code 8×128 | λ=0,095 | 54,35% | 0,005387 | — |
| instructie→code 8×128 | **δ=0,018** | **55,71%** | **0,005221** | discrete dominantie |
| WikiText 1×1.024 | λ=0,095 | 43,19% | 0,006063 | — |
| WikiText 1×1.024 | **δ=0,018** | 41,25% | **0,005505** | 1,94 pp minder load, 9,2% lagere KL |

De instructie/codepunten domineren de gekoppelde vaste prior op zowel
validatie als test. De lange-contexttest behoudt het lagere-KL-patroon, maar
heeft slechts één blok per split en daarom geen betrouwbaar interval.

Bij het agressievere WikiText-punt δ=0,016 is de relatieve test-CE `+0,331%`
met een 95%-interval net boven nul. Dat punt is daarom een frontpunt, geen
“gratis kwaliteit”-claim. δ=0,004 is de conservatieve onderzoekskandidaat.

## 4. Waarom routevrijheid aannemelijk is

Op alle 1.024 validatie- plus 1.024 testtokens van de laag-26-trace zijn alle
924 top-12-kies-6-routes exact uitgevoerd, met ongewijzigde DeepSeekgewichten.
Op de testset:

- heeft 85,64% van de tokens minstens één alternatief bij lokale KL ≤0,001;
- zijn er bij die grens gemiddeld 195,70 alternatieven en 5,043 bits
  route-equivalentie-entropie;
- heeft de beste alternatieve route gemiddelde KL 0,000427;
- heeft de beste route met Jaccard-overlap ≤0,5 gemiddelde KL 0,000922;
- zijn volledig disjuncte routes niet veilig: gemiddelde KL 0,08594.

Er is dus veel maar gestructureerde routevrijheid. Een massabudget is een
goedkope router-only proxy om daar conservatief gebruik van te maken; het is
geen bewijs dat iedere toegestane route functioneel equivalent is.

Bron: `baseline/layer26_route_equivalence_full.json`.

## 5. Falsificatie van de geleerde risicoselector

Een MLP kreeg alleen runtime-beschikbare routerfeatures, expert-ID's en een
32-dimensionale train-only PCA van de huidige MoE-input. Train, early-stop,
conformal calibration en test waren strikt gescheiden.

Op de onaangeraakte test is de log10-KL-RMSE `0,865` en Pearson `0,612`. Bij
α=0,05 dekt de simultane bound slechts 90,82% van de kandidaat-slates, niet de
beoogde 95%. Bij α=0,10 en risicolimiet 0,003 bespaart de selector 13,77% loads,
maar drie van 1.024 gekozen routes overschrijden de grens en de maximale
lokale KL is 0,02073. De worst-case conformal correctie α=0,005 haalt 100%
slate-dekking, maar staat bij limiet 0,003 geen enkele vervanging toe en bij
0,01 slechts 1,19% loadreductie.

Verdict: nuttige diagnostiek, geen inzetbare veiligheidscertificering. Dit
negatieve resultaat is niet gebruikt om de Mass-Budget-claim kunstmatig te
versterken.

Bron: `baseline/layer26_conformal_cache_selector_full.json`.

## 6. Autoregressieve tegenproef

In één prompt met vier greedy vervolgtokens gebruiken alle policy's een eigen
persistente expert-LRU en een gedeelde batched attention-KV-cache. Alle 4/4
gegenereerde tokens zijn gelijk, maar het kwaliteitspad domineert niet:

| Policy | totale loadreductie | pre-decision-KL per stap |
|---|---:|---|
| vaste λ=0,085 | 18,16% | 0,00390; 0,00650; 0,00794; 0,01584 |
| δ=0,012 | 14,01% | 0,00759; 0,00712; 0,00602; 0,01458 |
| δ=0,014 | 16,31% | 0,01266; 0,01539; 0,02021; 0,01733 |
| δ=0,016 | 19,13% | 0,00973; 0,01198; 0,02024; 0,02200 |

δ=0,016 bespaart iets meer loads maar heeft op iedere stap hogere KL dan de
vaste prior. δ=0,012 is soms beter in KL, maar bespaart duidelijk minder. Eén
korte prompt is onvoldoende om gemiddeld gedrag vast te stellen, maar wel
voldoende om **universele autoregressieve dominantie te falsificeren**.

Bronnen: `baseline/matched_cache_policy_kv_rollout_4tokens.json` en
`baseline/matched_cache_policy_kv_rollout_delta12_14_4tokens.json`.

## 7. Byte-accounting, niet latency

Eén routed expert bevat 8.650.752 parameters. In de gebruikte hypothetische
packed-int4-indeling, inclusief BF16-rijschalen, is dat 4,134 MiB per expert.
Op het vooraf vastgelegde testvenster bespaart δ=0,004 13.990 expert-loads over
2.048 tokens:

- strict projected routed I/O: 201,48 MiB/token;
- Mass-Budget projected routed I/O: 173,24 MiB/token;
- verschil: **28,24 MiB/token**.

Exploratieve agressievere punten projecteren 70,60 MiB/token minder op het
eerste WikiText-venster en 110,62 MiB/token minder op de code-test. Dit zijn
deterministische accountingcijfers. De evaluator draaide BF16 en mat geen
packed transfers, prefetchoverlap, kerneloverhead, throughput of energie.

Bron: `baseline/mass_budget_cache_accounting.json`.

## 8. Novelty-afbakening

Cache-Conditional Experts bevat al:

- Max Rank met gegarandeerde top-J;
- Cumsum, dat per token een dynamische maximumrang uit cumulatieve
  routerprobabiliteit afleidt;
- Cache-Prior, dat gecachete experts met een vaste λ herordent en de originele
  gewichten behoudt.

Mass-Budget combineert deze bekende bouwstenen anders: zij genereert een vaste
Cache-Prior-routefamilie, legt een expliciet verliesbudget op de **geselecteerde
top-6-probability mass** en minimaliseert daarbinnen de huidige misses. In een
gerichte zoekactie is geen paper gevonden met precies deze beslisregel. Dat is
geen uitputtend literatuur- of patentonderzoek. Cumulatieve-massagating,
residency-aware routing en counterfactual route-evaluatie zijn ieder al prior
art; de verdedigbare claim is dus **een empirisch sterke incrementele policy**,
niet een fundamenteel nieuw paradigma.

Zie `docs/PRIOR_ART.md` voor de volledige afbakening.

## 9. Beperkingen die de claim begrenzen

1. De preregistratie gebruikt een nieuw tokenvenster, maar nog steeds dezelfde
   WikiText-dataset en hetzelfde model.
2. Blokbootstraps behandelen 128-tokenblokken als sampling units; zij lossen
   alle sequentiële afhankelijkheid niet op.
3. De modelbrede hoofdevaluatie is teacher-forced taalmodellering, geen lange
   vrije generatie of MMLU/GSM8K-accuracy.
4. Alleen capaciteit 32, LRU, DeepSeek-V2-Lite Base en één modelcommit zijn
   sterk onderzocht.
5. De lokale instructie/codecorpora zijn relevante transferchecks, geen
   gestandaardiseerde benchmark.
6. Het probability-massbudget is geen formele bovengrens op output-KL.
7. De Pythonreferentie is niet geoptimaliseerd; een frontwinst kan door
   routeroverhead of systeemcomponenten worden opgegeten.

## 10. Definitief onderzoeksbesluit

De vooraf gestelde onderzoeksdoelstelling is gehaald: er is een policy die de
oude rank-7-baseline overtuigend verslaat en een beter kwaliteit-versus-load-
front laat zien dan vaste Cache-Prior op meerdere splits, met een vooraf
vastgelegde bevestiging en exact gecontroleerde modeluitvoering.

De aanbevolen volgende kandidaat is **`mass_budget:j2:0.004`**. Zij mag verder
naar lange multi-promptrollouts, MMLU/GSM8K en een packed runtime. Zij mag nog
niet als production-ready, formeel veilig of sneller in wall-clock worden
gepresenteerd. δ=0,018 blijft uitsluitend een agressief onderzoekspunt.

## Reproduceren

```powershell
# Unit tests
.\.venv\Scripts\python.exe -m pytest -q

# Volledige laag-26-route-equivalentie
.\.venv\Scripts\python.exe scripts\evaluate_layer26_route_equivalence.py `
  --tokens-per-split 1024 `
  --artifact-name layer26_route_equivalence_full.safetensors `
  --report-name layer26_route_equivalence_full.json

# Conformal falsificatie
.\.venv\Scripts\python.exe scripts\evaluate_layer26_conformal_cache_selector.py `
  --report-name layer26_conformal_cache_selector_full.json

# Vooraf vastgelegde confirmatie
.\.venv\Scripts\python.exe scripts\evaluate_modelwide_cache_routing_pareto.py `
  --blocks-per-split 16 --block-size 128 --token-offset 4096 --capacity 32 `
  --corpus-preset wikitext `
  --policy original --policy max_rank:j5:7 `
  --policy cache_prior:j2:0.0275 --policy mass_budget:j2:0.004 `
  --policy cache_prior:j2:0.095 --policy mass_budget:j2:0.018 `
  --report-name preregistered_wikitext_offset4096_mass_budget_confirmation.json

# Byte-accounting
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\build_mass_budget_cache_accounting.py

# Machine-audit van de vijf preregistratiegates
.\.venv\Scripts\python.exe scripts\verify_mass_budget_confirmation.py
```

Alle bron-JSON's staan in `reports/baseline/`; de index daar onderscheidt
authoritatieve, exploratieve en vervangen artefacten.
