# Research log

## 2026-08-09 — Projectstart en baselinebesluit

Besluit: beginnen met `deepseek-ai/DeepSeek-V2-Lite` Base. DeepSeek V4 Flash
blijft een latere schaaltest en Kimi K3 een north-star, geen ontwikkelmodel.

Waarom V2-Lite:

- officiële open teacher met 15,7B totaal en 2,4B actieve parameters;
- 26 MoE-lagen met 64 routed experts en top-6 routing;
- één expertlaag is afzonderlijk te onderzoeken binnen 64 GiB systeem-RAM;
- veel kortere iteratiecyclus en minder opslagrisico dan V4 Flash/K3.

Belangrijkste bedreiging voor validiteit: routerdrift. Een correctie met een
originele expert is niet globaal exact wanneer eerdere approximatieve lagen de
hidden state al hebben veranderd. Daarom zijn autoregressieve student-rollouts
een verplichte gate en geen optionele eindtest.

Status bij start:

- workspace was leeg en bevatte nog geen commits;
- Windows 11 Pro, Intel Core Ultra 9 285H;
- 63,43 GiB zichtbaar RAM;
- RTX PRO 2000 Blackwell Laptop GPU, 8.151 MiB VRAM;
- NVIDIA-driver 595.71, driver rapporteert CUDA 13.2;
- 310,97 GiB vrij op C:;
- Python 3.12.10 aanwezig; PyTorch/Transformers nog niet geïnstalleerd.

Volgende stap: omgeving installeren, officiële modelrevisie pinnen, hardware-
en synthetische baselinerapporten genereren.

## 2026-08-09 — Baseline 0 uitgevoerd

- Modelrevisie vastgezet op `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Checkpointmanifest declareert 31.413.626.576 bytes (29,256 GiB) weights.
- Zes initiële unit tests slagen; CPU- en CUDA-smoketest werken.
- Checkpointlayouttooling toegevoegd. Laag 1 zit volledig in shard 1, laag 13
  in shard 2 en laag 26 in shard 4. Daardoor kan onderzoek incrementeel starten.
- Traceformaat vastgelegd met hidden states, top-k ids/gewichten, individuele
  geselecteerde expertoutputs, routed aggregate en shared-expertoutput.

Belangrijk detail uit de officiële code: `norm_topk_prob` is `false`. De zes
geselecteerde routergewichten worden bij inference dus niet opnieuw tot som 1
genormaliseerd; ze zijn de geselecteerde waarden uit de softmax over alle 64
experts. Zowel een ongewogen als een gewogen top-1-baseline moet daarom worden
gerapporteerd.

## 2026-08-09 — Runtimecompatibiliteit

De eerste combinatie PyTorch 2.11.0+cu128, NumPy 2.5.2 en Transformers 5.14.1
faalde al bij import van `AutoTokenizer`: Torch-Dynamo rapporteerde ten onrechte
dat `NP_SUPPORTED_MODULES` uit een gedeeltelijk geïnitialiseerde module kwam.
Een exception trace op de door Torch-Dynamo ingeslikte `ImportError` vond de
werkelijke oorzaak: Windows Application Control blokkeerde NumPy's
`_bounded_integers`-DLL. PyTorch 2.12.1+cu132 met NumPy 2.2.6 en Transformers
5.14.1 doorstaat zowel `import torch._dynamo` als de officiële DeepSeek-config-
en tokenizerprobe. Dit is een omgevingsbevinding, geen modelresultaat.

## 2026-08-09 — Volledige checkpoint en echte-gewichtensmoke

- Alle vier BF16-shards zijn gedownload op de gepinde modelcommit; samen
  declareert de index 31.413.626.576 tensorbytes (29,256 GiB).
- Elf unit tests slagen met PyTorch 2.12.1+cu132 en NumPy 2.2.6.
- De laag-1-smoke laadt de 64 routed experts, router en twee shared experts met
  hun echte gewichten en voert de officiële SiLU-gated MLP-formule op CUDA uit.
- De top-6 aggregate is uit de bewaarde individuele expertoutputs
  gereconstrueerd met maximaal `6,005e-5` absolute afwijking.
- De som van de zes geselecteerde routergewichten is gemiddeld `0,182890`
  (minimum `0,149750`, maximum `0,220810`). Dit bevestigt empirisch dat de
  geselecteerde top-6 niet wordt herenormaliseerd.
- Op deze 64 niet-wetenschappelijke smoke-inputs scoort top-1 zonder
  renormalisatie NRMSE `0,728309`, cosine `0,703483`; top-1 met renormalisatie
  NRMSE `12,143953`, cosine `0,703483`.

Belangrijke beperking: de smoke-inputs zijn token-embeddings die rechtstreeks
aan laag 1 zijn gevoerd. Het zijn geen echte post-attention activaties. Deze run
valideert dus uitsluitend de loader, routing, expertformule, trace-opslag en
metriekketen; hij telt niet als bewijs voor of tegen compressie.

## 2026-08-09 — Eerste echte corpusbaseline en negatieve rankbevinding

WikiText-2-raw-v1 is vastgezet op datasetcommit
`b08601e04326c79dfdd32d625aee71d232d685c3`; alle drie parquetbestanden hebben
een lokale SHA-256 in `reports/baseline/wikitext_corpus.json`. De eerste vaste
steekproef bevat 2.048 tokens per officiële split in blokken van 128.

Top-1 zonder hernormalisatie geeft NRMSE `0,665569` op train, `0,643343` op
validatie en `0,639960` op test. De overeenkomst tussen de splits maakt het
onwaarschijnlijk dat de eerdere slechte top-1-score alleen door de lokale
smoketekst kwam.

Een gedeelde outputbasis met per-expert lineaire activatiecodes is getest op
ranks 4–256. Ridge wordt uitsluitend op validatie gekozen en test wordt daarna
eenmaal gerapporteerd. Bij rank 256 is de representatie `16,25×` kleiner dan de
BF16 routed bank, maar de test-NRMSE is `0,865493`: slechter dan top-1. Zelfs de
oracle die de echte expertoutputs perfect naar dezelfde basis projecteert haalt
slechts `0,762309`. Een aparte oracle-PCA op de al geaggregeerde routed output
haalt bij rank 512 NRMSE `0,667300`.

Interpretatie: de eenvoudige hypothese “de routed expertoutput leeft in één
kleine lineaire outputruimte” wordt op laag 1 niet ondersteund. De volgende
variant moet niet-lineair en route-conditioned zijn; alleen rank verhogen zou
de kernclaim niet bewijzen.

## 2026-08-09 — Niet-lineaire surrogates en quantisatiecontrole

Met 16.384 train-, 4.096 validatie- en 4.096 testtokens zijn twee niet-lineaire
families getraind. Een gedeelde SwiGLU met width 1408 haalt test-NRMSE
`0,853210` bij `64×` parameterreductie. Routergewichten als 64 extra features
verbeteren dit marginaal naar `0,852287`. Een gedeelde SwiGLU plus
expert-specifieke rank-256 inputadapters en een gedeelde outputbasis haalt
`0,842721` bij `12,96×`. Dit blijft slechter dan top-1 (`0,640528`).

De optimizerfailure bij learning rate `1e-3` is apart geïsoleerd: brede modellen
explodeerden na epoch 2. Met learning rate `3e-4` en gradient clipping bleef de
training stabiel, maar de wetenschappelijke conclusie veranderde niet.

Per-output-row fake quantization op de originele routed experts geeft:

- 8-bit: `1,998×`, test-NRMSE `0,009797`;
- 4-bit: `3,991×`, test-NRMSE `0,155886`;
- 3-bit: `5,317×`, test-NRMSE `0,374014`;
- 2-bit: `7,964×`, test-NRMSE `0,925630`.

Een vooraf vastgelegde mixed policy selecteert op train de 32 experts met de
hoogste routermassa, bewaart die op 3-bit en maakt de overige 32 binair. De beste
policy is op validatie gekozen en haalt op test `0,502063` bij `7,964×`. Dit
verslaat zowel uniforme 2-bit, top-1 als alle geteste surrogates.

## 2026-08-09 — Volledige-diepte teacher-forced evaluatie

Met 256 testtokens zijn alle 27 lagen vanaf de gepinde checkpoint één voor één
over de GPU gestreamd. Wanneer alleen laag 1 wordt vervangen:

- mixed quant: logit-NRMSE `0,052368`, KL `0,008302`, top-1-overeenkomst
  `0,964844`;
- rank-256 surrogate: logit-NRMSE `0,124023`, KL `0,035795`,
  top-1-overeenkomst `0,933594`.

Vervolgens zijn per laag op 2.048 treintokens de 32 hete experts gekozen en
zijn alle 26 routed banken met dezelfde 3/1-bit-policy uitgevoerd. De run is
exact reproduceerbaar over twee herhalingen. Resultaat: late routeroverlap daalt
tot ongeveer `0,75–0,83`, eindlogit-NRMSE is `0,340373`, KL `0,907879`,
top-1-overeenkomst `0,683594` en next-token cross-entropy stijgt van `2,328192`
naar `3,268534` (`+0,940342`).

Exacte header-accounting: 91,649% van de checkpointtensorbytes zit in routed
experts. De policy reduceert die banken `7,964×`, maar de volledige checkpoint
slechts `5,036×`, tot `5,810 GiB`. De 8× routed-bankvariant faalt dus de
modelkwaliteitsgate door cumulatieve fout en routerdrift.

De volledige Pareto-ablaties geven:

- uniform 4-bit over alle routed banken: logit-NRMSE `0,163098`, KL `0,150721`,
  top-1-overeenkomst `0,914063`, cross-entropy `+0,114595`;
- uniform 3-bit: logit-NRMSE `0,265539`, KL `0,493746`, top-1-overeenkomst
  `0,765625`, cross-entropy `+0,560174`;
- hot-3/cold-1 (~2 gemiddelde bits): logit-NRMSE `0,340373`, KL `0,907879`,
  top-1-overeenkomst `0,683594`, cross-entropy `+0,940342`.

Ook 4-bit heeft dus nog circa 4,9% relatieve cross-entropyverslechtering op de
kleine teacher-forced steekproef. De vooraf gewenste grens van minder dan 2%
is nog niet gehaald. Lagen 24–26 versterken de hidden-statefout opvallend sterk
bij zowel uniforme 3- als 4-bit quantisatie; een gelijke precisie per laag is
daardoor waarschijnlijk suboptimaal.

Een exploratieve laagallocatie is op validation vergeleken. Alleen de laatste
drie of zes lagen op 8-bit helpt minder dan verwacht; die lagen versterken dus
vooral eerder opgebouwde afwijkingen. De beste policy bij hetzelfde aantal
opgewaardeerde lagen zet zowel lagen 1–3 als 24–26 op 8-bit en de middenlagen op
4-bit. Op validation is de cross-entropytoename `+0,062456` (`+3,21%`). Op een
nog niet eerder gebruikt testsegment (blokken 2–3) is zij `+0,051176`
(`+2,32%`), met KL `0,064319` en top-1-overeenkomst `0,933594`. Dit benadert de
2%-gate, maar haalt haar niet en offert ook routed-bankcompressie op.

Exacte storage-accounting voor deze edge-policy: `3,244×` reductie van de
routed banken, `2,732×` voor de volledige checkpoint en een geraamde grootte van
`10,708 GiB`. De policy haalt de vooraf vereiste 4× routed reductie dus niet.

Twee korte echte greedy rollouts zijn zonder KV-cache gestreamd. Op de
feitelijke prompt blijven acht tokens identiek; op een open prompt vier tokens.
In totaal zijn 12/12 teacher/studenttokens gelijk, maar de gemiddelde
top-6-routeroverlap voor een nieuw token varieert van `0,782051` tot `0,974359`.
Dit is een geslaagde stabiliteitssmoke, geen bewijs voor honderden tokens of
brede promptkwaliteit. De referentieruntime kost circa 63–99 seconden per token
door herhaald streamen van de BF16-checkpoint en is geen deploymentbenchmark.

## 2026-08-09 — Hybrid quantization-residual

Als laatste eigen vervolgvariant is uniform 4-bit behouden en alleen het
activatieresidu gecorrigeerd met een shared outputbasis plus expert-specifieke
lineaire codes. Op laag 1 geeft 4-bit alleen NRMSE `0,155886`. Rank 128 verbetert
naar slechts `0,150087`, terwijl routed compressie daalt van `3,991×` naar
`3,554×`; de oracle haalt `0,144244`. De quantisatiefout is dus eveneens te
hoog-rank/onvoorspelbaar voor deze lineaire corrector.

Definitieve afgebakende conclusie: de geteste eenvoudige shared-basis,
aggregate-SwiGLU en expert-low-rank activation-surrogates zijn een no-go voor de
vooraf gestelde V2-Lite-gates. Laagbewuste quantisatie is duidelijk kansrijker,
maar haalt tegelijk noch 4× routed reductie noch <2% cross-entropyverlies. Er is
geen rechtvaardiging om nu V4 Flash te downloaden.

## 2026-08-09 — QuotientQuant, eerste falsificatietest

Voor laag 26 is op 2.048 WikiText-treintokens de horizon-0
future-logit-Fishermatrix geschat. De exacte BF16-laaguitgang is vergeleken met
dezelfde laag waarvan alleen alle routed experts per outputrij naar 3-bit zijn
gequantiseerd. De basis is uitsluitend op train bepaald; validation en test
zijn onaangeraakt.

De eerste Monte-Carlo-run met één Fisher-score per state geeft participation
rank `222,24`. Rang 64 vangt `36,81%` van de geschatte behavioral sensitivity;
zelfs rang 256 vangt maar `71,34%`, zodat `r90 > 256`. Op test herstelt de
geprojecteerde oracle bij rang 64 `17,06%` van de KL-schade en `22,06%` van de
CE-schade. Bij rang 256 is dit respectievelijk `38,54%` en `49,18%`.

De behavioral basis verslaat meestal error-PCA en random, dus het signaal is
reëel. De vooraf vastgelegde Eureka-gates (`80–90%` sensitivity en minstens
`70%` schadeherstel bij rang 64) worden echter ruim gemist. Omdat een enkele
Fisher-trekking het spectrum kan vertekenen, volgt een achtvoudige herhaling,
volledige eigendecompositie en split-half-stabiliteitscontrole voordat deze lijn
definitief wordt verworpen.

De achtvoudige herhaling bevestigt de no-go: `r80 = 609`, `r90 = 943` en
participation rank `260,21`. Op test herstelt rang 64 nog maar `13,30%` KL en
`18,60%` CE; rang 512 herstelt `33,90%` KL. Zelfs een direct op schadelijke
trainfouten gewogen error-PCA haalt bij rang 64 slechts `11,30%` KL-herstel.
Daarmee wordt geen echte projected weight patch gebouwd: het gemeten
oracleplafond ligt al ver onder de gate en de benodigde rang vernietigt het
opslagvoordeel.

## 2026-08-10 — Dynamic-precision-oracles en predictor-falsificatie

Op laag 26 zijn alle 64 mogelijke 3→4-bit-upgrademasks per token met exacte
eindlogit-KL geëvalueerd. Het perfecte testoracle heeft 23,812% upgrades nodig
om de all-4-bit-KL binnen 1% te evenaren. Bij 25% upgrades is KL `0,003193`,
tegen `0,003245` voor volledig 4-bit. Er bestaat dus echte, tokenafhankelijke
precisiesparsity.

De inzetbare voorspellers halen dit plafond niet. Routerfeatures, lokale
outputfouten, een kwadratisch maskmodel en progressieve bitplane-signalen
blijven bij 25% duidelijk boven de all-4-bit-KL. Zelfs met de ware q4-delta als
Euclidische lokale score blijft test-KL circa `0,00536`. Het 2→4-bit-oracle
heeft 42,546% upgrades nodig en mist daarmee de vooraf gestelde 20–25%-gate.

Besluit: het 3→4-oracle is een positieve bovengrens, maar er is geen bruikbare
teacher-free selector bewezen. Deze richting wordt niet als Eureka geclaimd.

## 2026-08-10 — Route-equivalence entropy bevestigd

Voor 256 validatie- en 256 testtokens in laag 26 zijn alle 924 top-6-subsets uit
de twaalf hoogste routerkandidaten exact doorgerekend. De originele, niet
hergenormaliseerde DeepSeek-routerweights zijn behouden; er is geen coefficient
fit gebruikt.

Op test heeft bij KL ≤ `0,001` 85,547% van de tokens minstens één alternatieve
route, gemiddeld 202 alternatieven en gemiddeld `5,087` bits
route-equivalence entropy. De beste alternatieve route heeft gemiddelde KL
`0,000427` (p95 `0,001786`). Zelfs met Jaccard-overlap ≤ 0,5 is de beste route
gemiddeld `0,000974` KL. Volledig disjuncte routes zijn daarentegen schadelijk:
gemiddelde KL `0,141261`. De equivalentie is dus groot maar niet willekeurig.

Een interventie in laag 23, gevolgd door exacte lagen 24–26, bevestigt dat het
effect downstream overleeft. De volledig teacher-free regel die routerrangen
5–6 door 7–8 vervangt geeft test-KL `0,001396`, top-1-overeenkomst `99,22%` en
relatieve CE-delta `+0,112%` op 256 tokens.

## 2026-08-10 — Modelbrede cache-aware reproductie

De conservatieve modelbrede policy bewaart top-5 en vervangt rang 6 uitsluitend
door rang 7 wanneer dit, gegeven de huidige studentcache, exact minder
within-token LRU-misses geeft. De kwaliteit wordt in BF16 gemeten; de strict
baseline gebruikt de ongewijzigde teacherroutes en een onafhankelijke cache.

Met capaciteit 32 per MoE-laag en acht onafhankelijke blokken van 128 tokens per
split:

- WikiText, 2.048 tokens totaal: 92.607 → 82.221 expert-loads (`−11,215%`);
  test-KL `0,003861`, top-1 `97,36%`, relatieve CE `+0,097%`. De 95%-bootstrap-
  intervallen op de acht testblokken zijn `10,09–11,28%` loadreductie en
  `−0,130–+0,328%` relatieve CE-delta.
- Nederlandse instructie/researchtekst plus Pythoncode, 2.048 tokens totaal:
  95.762 → 83.797 loads (`−12,495%`); codetest-KL `0,002614`, top-1 `98,14%`,
  relatieve CE `−0,082%`. De testintervallen zijn `12,16–14,86%` loadreductie
  en `−0,325–+0,213%` relatieve CE-delta.

Alle 32 afzonderlijke 128-tokenblokken over beide corpora/splits hebben een
positieve loadreductie. CE-verandering is in iedere split statistisch met nul
verenigbaar; KL is klein maar niet nul.

Een echte KV-cache-smoke met onafhankelijke greedy teacher/studentprefixes
produceert 4/4 dezelfde tokens. Pre-decision-KL stijgt van `0,00795` naar
`0,01872`; over de hele run dalen loads 1.343 → 1.283 (`−4,47%`). Enkele losse
stappen hebben door verschillende cachehistorie juist méér studentloads. Dit is
een stabiliteitssmoke, geen lange-rolloutbewijs.

## 2026-08-10 — Byte-accounting en definitieve afbakening

Eén routed expert bevat 8.650.752 parameters: 16,50 MiB in BF16 of 4,134 MiB in
de gebruikte hypothetische packed-int4-accounting inclusief BF16-rijschalen.
Capaciteit 32 per elk van 26 lagen vraagt daarom 13,406 GiB in BF16 of
3,359 GiB in int4. Samen met de niet-routed BF16-weights is dat in het int4-
cachescenario 5,802 GiB resident; niet-gecachete experts moeten dan uit een
tragere tier komen.

De gemeten loadreductie projecteert bij int4 naar 20,97 MiB minder routed I/O
per WikiText-token en 24,15 MiB minder per instructie/code-token. Dit zijn
accountingcijfers, geen kernel- of wall-clockmetingen. De evaluator voerde BF16
uit; combineren met int4 moet opnieuw end-to-end worden gevalideerd.

De gerichte novelty-audit vindt dat de praktische policy vrijwel het
`J=5, M=7`-grensgeval is van Max Rank uit Cache-Conditional Experts (TMLR 2025).
MoE-ERAS, BuddyMoE, SERE en SliceMoE bezetten de bredere claims rond
residency-aware selectie, functionele vervanging en dynamic precision. Een
paper uit mei 2026 test bovendien al sampled counterfactual routes op
DeepSeek-V2-Lite.

Definitief oordeel voor deze onderzoeksronde:

- de low-rank behavioral-observabilitypatch faalt haar gates;
- dynamische precisie heeft een oracle-effect maar geen werkende predictor;
- functionele route-niet-uniciteit is overtuigend lokaal en downstream bewezen;
- de eenvoudige cachetoepassing reproduceert een bruikbaar, stabiel effect;
- dit is geen nieuwe fundamentele uitvinding en nog geen deploymentbenchmark.

## 2026-08-10 — Primaire Cache-Prior-baseline gereproduceerd

De paper-faithful modelbrede evaluator laat alleen de gate-ID's afwijken en
voert verder de volledige officiële DeepSeek-decoderlaag en MoE-kernel uit. De
`original`-control heeft in alle 26 MoE-lagen maximale fout `0` en eind-KL/CE
`0`. Met contextlengte 1.024 en capaciteit 32 daalt de validatiemiss bij
Cache-Prior λ=0,5 van `26,29%` naar `6,53%`; op test van `31,50%` naar `8,48%`.
De relatieve test-CE stijgt `1,265%`. Dit reproduceert het gepubliceerde signaal
van ongeveer `28%→7%` en `0,1–3%` perplexiteitstoename voldoende nauwkeurig.

De conservatieve vaste prior λ=0,025 verslaat de oude Max-Rank J5/M7 al op KL
bij vergelijkbare of hogere loadreductie. Cache-Prior, niet rank-7, is daarom
vanaf hier de serieuze baseline.

## 2026-08-10 — Volledige route-equivalentietrace

De laag-26-enumeratie is uitgebreid naar alle 1.024 validatie- en 1.024
testtokens. Bij KL ≤`0,001` heeft 85,64% van de testtokens een alternatief,
gemiddeld 195,70 alternatieven en 5,043 bits route-equivalentie-entropie. De
beste alternatieve route heeft gemiddelde KL `0,000427`; volledig disjuncte
routes blijven schadelijk met gemiddelde KL `0,08594`. De eerdere conclusie
over veel maar gestructureerde routevrijheid blijft dus op de volledige trace
staan.

## 2026-08-10 — Held-out risicoselector gefalsificeerd

Een MLP met routerfeatures, expert-ID's en een train-only 32D-PCA is gescheiden
getraind, vroeg gestopt, conformal gekalibreerd en op 1.024 onaangeraakte tokens
getest. Test log10-KL-RMSE is `0,865`, Pearson `0,612`. Bij α=0,05 is de
simultane kandidaat-slate-dekking slechts `90,82%`, onder de beoogde 95%.
Worst-case α=0,005 herstelt 100% dekking, maar levert bij KL-limiet `0,003`
geen loadwinst en bij `0,01` slechts `1,19%`.

Besluit: de predictor is diagnostisch, niet inzetbaar als veiligheidsbewijs.
Deze negatieve richting wordt gesloten voor het huidige Eureka-verdict.

## 2026-08-10 — Mass-Budget Cache-Prior ontwikkeld

Nieuwe trainingvrije policy: bewaar top-2, genereer dertien vaste
Cache-Prior-routes, verwerp routes waarvan de som van originele geselecteerde
routerkansen meer dan δ onder de originele top-6 ligt, en kies vervolgens de
route met de minste actuele LRU-misses. Expertoutputs blijven door de originele
routerkansen gewogen.

Exploratief verbetert δ=0,004 het discrete front op WikiText en bij
instructie→codetransfer. Op code domineert δ=0,018 vaste λ=0,095 tegelijk in
load (`55,71%` versus `54,35%`) en KL (`0,005221` versus `0,005387`). Een
1.024-token-context behoudt het lagere-KL-patroon. Het agressievere
WikiTextpunt δ=0,016 heeft wel een kleine positieve CE-regressie en wordt niet
als gratis winst geclaimd.

## 2026-08-10 — Vooraf vastgelegde confirmatie: alle gates geslaagd

Vóór uitvoering zijn offset 4.096, 16×128 tokens per split, alle policy's en
vijf gates vastgelegd. Op dit nieuwe WikiTextvenster verslaat δ=0,004 de oude
rank-7-regel op test in zowel load (`14,017%` versus `10,559%`) als KL
(`0,003704` versus `0,004389`). Tegen λ=0,0275 is de KL op validatie en test
15,54% en 15,11% lager, met slechts 1,379 en 1,393 procentpunt minder
loadreductie; dit valt binnen de vooraf bepaalde marge van 2 punten.

Relatieve test-CE voor δ=0,004 is `−0,057%`, 95%-blokbootstrap
`−0,171%–+0,060%`. Alle zestien testblokken besparen loads en de exact-control
blijft foutloos. Er is niets na inspectie herafgesteld. Dit is de sterkste
bevestiging van de huidige policy.

## 2026-08-10 — Autoregressieve grens en byteprojectie

Een vier-token-KV-rollout houdt voor alle policy's 4/4 tokenovereenkomst, maar
falsificeert universele dominantie. δ=0,016 bespaart `19,13%` loads tegenover
`18,16%` voor λ=0,085, maar heeft op alle vier stappen hogere pre-decision-KL.
Conservatievere δ's ruilen kwaliteit en load anders uit en domineren evenmin.

In packed-int4-accounting is één expert 4,134 MiB. Op de prereg test bespaart
δ=0,004 13.990 loads over 2.048 tokens, oftewel een geprojecteerde 28,24 MiB
routed-weight-I/O per token. Dit is geen gemeten latency of int4-uitvoering.

Eindoordeel: een begrensde praktische Eureka is bereikt voor het modelbrede
teacher-forced kwaliteit-versus-loadfront. Production-speed, formele
outputveiligheid en algemene autoregressieve winst blijven open.

## 2026-08-10 — CRAFT H7 sparse-route-coresetgate niet gehaald

Na preregistratie zijn op laag 26 alle subsets van de zes reeds geselecteerde
expertoutputs uitgeput met vrije LS, exacte NNLS, box-begrensde LS en vaste-
routergewichtcontroles. De officiële top-6-deltacontrol is exact.

Bij lokale teacher→candidate-KL ≤`0,001` is de hogere empirische mediaan van de
minimumcardinaliteit `4` en p95 `6` op zowel 256 validatie- als 256 testtokens.
Daarmee faalt de positieve H7-gate (`mediaan≤3` of `p95≤4`). De harde
falsificatie triggert niet: bij NNLS `k=5` blijft slechts `3,91%` validatie en
`2,34%` test boven KL `0,003`. Vrije LS en NNLS hebben dezelfde curve; negatieve
coëfficiënten lossen de cardinaliteitsstaart dus niet op.

Besluit: H7 stopt vóór de laag-23-interventie en blijft als
`inconclusive_negative` resultaat bewaard. De volgende onafhankelijke P0-oracle
mag openen zonder H7 op testdata bij te stellen.

## 2026-08-10 — CRAFT H1 CRCQ top-32-screen sterk positief

Na preregistratie zijn per token alle 924 all-Q3-routes met exacte LM-head-KL
gemeten en op de 32 beste routes alle 64 Q3/Q4-maskers. Tegen hetzelfde
`1,01×`-all-Q4-doel heeft de natuurlijke route 20,313% validatie- en 22,461%
testupgrades nodig. Gezamenlijke route+bitselectie verlaagt dit naar **11,263%**
en **14,128%**, oftewel 44,55% en 37,10% relatieve upgradereductie. De vooraf
vastgelegde 15%-puntgate slaagt op beide splits.

All-Q3-rerouting alleen sluit 51,65% van de KL-gap op validatie maar slechts
25,15% op test. Het gerepliceerde signaal is daarom specifiek de joint
route×bitvrijheid. De test-blockbootstrap voor de joint upgradefractie is breed
(`12,50–17,12%`) en kruist de gate; dit is een sterke exploratieve oracle, geen
confirmatie of deploybare selector.

De preregistratie staat nu een afzonderlijke volledige 59.136-kandidatenoracle
toe. De top-32-screen en zijn ruwe 55,9 MiB JSON blijven ongewijzigd bewaard.

## 2026-08-10 — CRAFT H1 volledige CRCQ-oracle positief

De vooraf vastgelegde volledige `924×64` route-maskerruimte is zonder retuning
geopend: 30.277.632 exacte volledige-vocabulaire-KL-metingen. De minimum-
upgradefractie daalt verder van de top-32-waarden naar **9,831%** op validatie
en **12,240%** op test, tegenover 20,313% en 22,461% voor natural-route-DP.
Dat is 51,60% en 45,51% relatieve reductie in Q4-upgrades bij hetzelfde
`1,01×`-all-Q4-KL-doel.

De directe gekozen schedule reproduceert de DP binnen `3,51×10⁻⁸`/exact nul.
Test relatieve CE-delta is `−0,010%`, validatie `+0,528%`, en top-1 is op beide
98,44%. De test-blockbootstrap voor upgradefractie is `10,677–15,169%` en raakt
de 15%-grens nog net; dit blijft exploratief.

Besluit: de lokale CRCQ-oracle op laag 26 is bewezen positief. Laag 23 plus
exacte lagen 24–26 mag nu worden geopend. Een deploybare selector, fysieke
runtime en confirmatory venster bestaan nog niet.

## 2026-08-10 — CRAFT H1 CRCQ downstream gefalsificeerd

Op laag 23 zijn opnieuw alle 59.136 route×bitkandidaten per token uitgeput,
ditmaal met lokale routed-output-MSE-selectie, waarna de gekozen sequence exact
door lagen 24–26 liep. Om lokaal natural-all-Q4 te evenaren heeft joint nog
75,521% validatie- en 70,508% testupgrades nodig; natural zelf 85,156% en
80,534%. De vooraf vastgelegde 15%-gate en 25%-falsificatiegrens worden ruim
gemist.

Bij 15% upgrades verbetert joint finale KL op validatie (`0,004976` versus
`0,005233` natural) maar verslechtert op test (`0,005088` versus `0,005011`).
De richting reproduceert niet. De BF16-control en alle exacte-tailcontroles
zijn foutloos.

Besluit: H1 blijft een sterk lokaal laag-26-oracleplafond, maar de vereiste
eerdere interventie is gefalsificeerd. Geen selector- of confirmatorywerk voor
H1; de P0-volgorde gaat door naar H3 atomic expert oracle.

## 2026-08-10 — CRAFT H3 atomaire laag-26-oracle positief

Na preregistratie zijn de 8.448 routed SwiGLU-atomen per token exact ontbonden
en met zes vaste neuron-/tegelselectors over acht retentiefracties geëvalueerd.
De globale bijdragescore behoudt 25% met relatieve CE `+0,1118%` validatie en
`−0,1390%` test; bij 10% is dit `+0,4086%` en `+0,0079%`. Daarmee slagen zowel
de 2%-primaire als 3%-moonshotgate op beide vooraf vastgelegde splits. De
gemiddelde KL is bij 25% `0,001106/0,001425` en bij 10%
`0,003954/0,005716` (validatie/test).

De tile-64-hardwaregate faalt: bij 25% is haar KL 8,63×/8,48× de globale
neuronoracle, ruim boven de limiet 1,20×. Ideale support-known bytes/MACs zijn
25%, maar tensor-lokale 4-KiB-paginadruk blijft voor globale atomen 50%. De
100%-teacher-deltacontrol is exact; alle masks en raw metrics staan in de
39,9-MB JSON.

Besluit: een sterk laat-laags oracleplafond, nog geen deploybare Eureka. H3
blijft open voor de vooraf vereiste laag-23-interventie met exacte lagen 24–26;
geen predictor of runtimeclaim vóór die downstreamproef.

## 2026-08-10 — CRAFT H3 laag-23 downstream positief

De vooraf gekozen globale atoomscore is op laag 23 toegepast en vervolgens
door officiële lagen 24–26 gevoerd. Bij 25% retentie is finale relatieve CE
`+0,0539%` validatie en `+0,2942%` test, KL `0,000906/0,000998` en top-1 op
beide `99,22%`. Bij 10% is relatieve CE `+0,2140%/+0,3250%`, KL
`0,001927/0,001800` en top-1 `99,22%/98,83%`. Primaire, moonshot- en alle
veiligheidsgates slagen.

De 25%-routeroverlap blijft na elke downstreamlaag minstens 98,96% op
validatie en 99,15% op test. De 100%-control blijft finale KL/CE nul. Ideale
atom-bytes zijn 25%, maar bestaande tensorpagina-accounting blijft 50%; er is
geen sparse runtime gemeten.

Besluit: H3 overleeft de verplichte eerdere interventie en mag naar de vaste
spread-layers 1/13/26 plus instructie/code-transfer. Dit is nog een
exact-supportoracle, geen inzetbare of modelbrede Eureka.

## 2026-08-10 — CRAFT H3 spread-layers en domeinen volledig positief

De vaste globale atoomscore is onafhankelijk op lagen 1, 13 en 26 getest over
WikiText-validatie/test, drie lokale instructieattachments en een bevroren
Pythoncorpus. Alle twaalf 25%-cellen halen CE `<2%`, KL `≤0,01`, top-1 `≥95%`
en de exact-control. Ook alle twaalf 10%-CE-moonshotcellen blijven onder 3%.

Het moeilijkste primaire punt is laag 1 op lokale instructies: KL `0,007461`,
relatieve CE `+0,1481%`, top-1 `96,88%`. Bij 10% wordt dezelfde cel kwetsbaar
met KL `0,01450` en top-1 `92,58%`, hoewel CE slechts `+0,1451%` is. Dat wordt
niet weggepoetst: 10% is nog geen uniforme veilige bedrijfsclaim.

Besluit: laag- en domeintransfer van het oracleplafond is aangetoond. H3 mag
naar de beslissende gelijktijdige full-depth-proef; afzonderlijke lagen tellen
nog niet als modelbrede Eureka of deploybare supportselectie.

## 2026-08-10 — CRAFT H3 gelijktijdig full-depth gefalsificeerd

Alle acht fracties zijn gelijktijdig in de 26 MoE-lagen uitgevoerd, waarbij
iedere policy haar eigen hidden states, routes en exacte supports volgde. De
25%-policy mist de primaire gate: WikiText-test relatieve CE is `+2,1129%`
tegen `<2%`; lokale instructies hebben KL `0,03505` tegen `≤0,03`. Validatie
zit met `+1,9229%` eveneens zonder robuuste marge. De 10%-moonshot faalt breed
met `+10,759%/+6,964%/+5,057%` CE op validatie/test/instructies.

Uniform 35% is het eerste curvepunt dat alle kwaliteitsgrenzen haalt, maar
levert slechts 2,86× ideale atomreductie en circa 56,7% tensor-lokale
paginadruk. Dat mag de gefaalde ≤25%-gate niet vervangen. De 100%-policy bleef
in alle 26 lagen exact; 208 lossless supporttensors bewaren het volledige raw
bewijs.

Besluit: H3 wordt gesloten als `falsified` voor de vereiste modelbrede 25%-
claim. De positieve lokale/downstreamresultaten blijven staan, maar H5/H9 en
een atomic packed runtime blijven geblokkeerd. De P0-volgorde gaat door naar
H4 Residual Syndrome Sketch.

## 2026-08-10 — CRAFT H4 SketchGate trace-anchored gefalsificeerd

De eerste H4-run miste de inhoudelijke gates, maar ook een extra vastgezette
Q3/Q4-herberekeningscontrol. Dat bestand is behouden. Een afzonderlijk
gepreregistreerde replicatie op splitposities 256–511 gebruikte daarom de
bevroren Q3/Q4-componentoutput bitexact en bouwde iedere kandidaat rechtstreeks
als `post_attention + (routed + shared)`. Routes en officiële teachercontrol
sloten exact.

De validatie koos zonder testinzage `r=64` Rademacher. Over vijf seeds is de
minimum oracle-recovery 83,74% op validatie en 82,18% op test; dit haalt de
80%-grens. De high-damage-FN is echter 22,73% en 24,68% voor de primaire seed,
ruim boven maximaal 1%. Down-only verklaart slechts 53,76%/65,89% van de
volledige oraclewinst, terwijl gate+up 79,86%/74,30% haalt. Metadata slaagt met
0,08345 effectieve bit, maar eager sketchcompute is 24,30% van de gemeten
vermeden transfertijd en mist de <10%-hardwaremodelgate.

Besluit: H4 is met sluitende controls gefalsificeerd. Geen spread naar lagen
13/23, geen OOD of complexere first-order sketch. De P0-volgorde gaat door naar
H2 Block-Coalescing Oracle.

## 2026-08-10 — CRAFT H2 Block-Coalescing hard gefalsificeerd

Op 8-tokenverificatieblokken zijn per token maximaal 32 exact-KL-eligible
top12-choose-6-routes gezamenlijk geoptimaliseerd. De exacte binaire ILP
reduceert de natuurlijke expert-unie slechts 19,65% op validatie en 20,24% op
test, tegenover de vaste 40%-gate en zelfs onder de 25%-harde grens. Extra
tegenover sequentieel Mass-Budget `δ=0,004` is de reductie 17,75%/18,24%, niet
de vereiste 25%. Gemiddelde lokale KL blijft veilig op
`0,000183/0,000200`.

Beam-1024 matcht de exacte unioncount in alle 64 primaire blocks. Alle 1.280
HiGHS-ILP's zijn optimal; een append-only audit corrigeert een adjudicator die
vier machine-epsilon MIP-gaps ten onrechte als niet exact nul behandelde.
Objective en gekozen union sluiten binnen `1,35×10⁻¹²`. Zelfs drempel 0,003,
cap 64 en 16-tokenblocks komen in de beamdiagnostiek slechts op circa
29,0–30,4% reductie.

Besluit: het combinatorische layer-26-oracleplafond is te laag. H2 stopt vóór
laag 23, bitplanes, atomtiles en speculative runtime. De P0-rij is uitgeput; de
volgende onafhankelijke hypothese is H6 QERC.

## 2026-08-10 — CRAFT H6 QERC in fase A gefalsificeerd

De vooraf geregistreerde decompositie van de zes gewogen Q3-expertfouten geeft
een globale cancellation fraction van `−1,129%` op validatie en `−0,106%` op
test. Beide absolute waarden liggen onder de vaste 2%-near-zero-grens; de
positieve kruistermsommen betekenen zelfs lichte foutversterking. De
teacher-delta-control is bitexact, expert-ID's en slotvolgorde zijn exact en de
routergewichtfout is nul.

Daarmee is de harde fase-A-stop getriggerd. Er zijn bewust geen per-row gains,
clipping- of floor/ceil-varianten gefit: die waren alleen toegestaan wanneer
natuurlijke co-routecancellatie de mechanistische preconditie haalde. De
same-byte schaalboekhouding blijft analytisch nul extra bytes, maar zonder
kandidaat of fysieke Q3-kernel volgt daar geen runtimeclaim uit.

Besluit: H6 is falsified voor de geregistreerde QERC-mechaniek. Ruwe covariance,
lossless slotcomponenten, hashes en het volledige stopbesluit zijn behouden. De
volgende onafhankelijke hypothese is H8 Cache Span Minimization.

## 2026-08-10 — CRAFT H8 Cache Span inconclusief negatief

De optimistische layer-26 ghost-cache-oracle kende de ware ontbrekende
expertoutput, koos per token de beste miss-subset na exacte vocabulaire-KL en
behield voor toekomstige tokens zelfs gratis de baseline-cachetoestand.
Validatie koos zonder testinzage bounded reconstructie uit geselecteerde
cachehits. Toch worden slechts `41,35%` validatie- en `48,54%` testloads
vermeden, niet de vereiste 50% op beide splits. KL blijft veilig op
`0,000251/0,000278`; er zijn nul extra expertforwards.

De beslissende ablation is zwakker: zero-fill vermijdt al `39,90%/48,31%`.
De lineaire span voegt dus slechts `+1,442/+0,225` procentpunt toe tegenover de
vooraf geëiste +10 punten. De twee-blockbootstrap is respectievelijk
`40,81–41,97%` en `45,08–51,19%` voor primary, maar met slechts twee blokken en
een gefaalde validatiepuntgate opent dit geen vervolg. Routes, loadcounts en de
official-original-control sluiten exact; alle raw kandidaten en een aparte
append-only bootstrapaudit zijn behouden.

Besluit: H8 sluit als `inconclusive_negative`, zonder predictor, causale cache,
laag-23-proef of full-depth. De volgende onafhankelijke richting is H10
Reduction Order; atomic-index/BiSparse/runtime blijven door H3 geblokkeerd.

## 2026-08-10 — CRAFT H10 Reduction Order hard gefalsificeerd

Voor Q3 en Q4 zijn alle 720 slotordes onder acht vaste sequentiële/boom- en
FP16/BF16/FP32-semantieken geëvalueerd. Validatie koos BF16-sequentieel met
orde `[3,5,1,4,0,2]`. Die vaste orde sluit slechts `1,487%` van de Q3→Q4-
KL-kloof op validatie en `0,829%` op test, ruim onder de vooraf vastgelegde
20%-gate én de 10%-harde stop. De gepaarde twee-blockintervallen zijn
`0,924–2,261%` en `0,672–0,886%`; geen bootstrapresample bereikt 10%.

De validation-gekozen FP32-control levert circa nul closure. BF16-termen die
exact naar FP32 worden gepromoveerd zijn voor Q3/Q4 en beide splits volledig
orde-invariant. Zelfs de per-token lokale MSE-oracle sluit exact slechts
`0,831%/1,183%` KL-gap. Routes, weights en official-original-control sluiten
exact; alle raw 8×720×256-series staan lossless in safetensors.

Besluit: H10 is hard gefalsificeerd en stopt vóór reducerbenchmark, laag 23,
spread en full-depth. Alle niet-geblokkeerde technische hypothesen uit het
huidige CRAFT-pakket zijn nu gescreend; de volgende stap is een onafhankelijke
reproducibility/claims-eindaudit in plaats van post-hoc varianten.

## 2026-08-10 — CRAFT reproduceerbaarheidsaudit geslaagd

De append-only verifier heeft 751 onafhankelijke controles uitgevoerd: 748
geslaagd, nul verplichte fouten en drie expliciete waarschuwingen. Alle
technische gates zijn opnieuw uit numerieke metriekvelden berekend. H7 is uit
per-token-NNLS-KL herbouwd, H8 is opnieuw uit twaalf validatieconfiguraties
geselecteerd en H10 is rechtstreeks uit het lossless `8×720×256`-sweepbestand
opnieuw geminimaliseerd. Beide H8/H10-sequence-blockbootstraps sluiten exact.

Het 136-entry SHA-256-manifest sluit met alle eerder gepubliceerde en intern
gedeclareerde hashes. De falende H4-eerste run en foutieve H2-audit-v1 zijn
behouden; audit-v2 bevestigt alle 1.280 exacte ILP-records. De drie
waarschuwingen begrenzen uitsluitend claims: geen thermal/clock-telemetrie voor
H4/H8 en geen numerieke definitie van H10’s kwalitatieve Q4-“catastrophic”-term.

Besluit: de bestaande positieve tussenoracles en negatieve eindbesluiten zijn
correct gerapporteerd. Dit is bewijs van reproduceerbaarheid, niet van Eureka of
snelheid. De afgescheiden novelty-/claims-audit blijft de laatste voorwaarde
voor het masterverdict.

## 2026-08-10 — CRAFT onafhankelijke novelty-audit gesloten

De vooraf vastgelegde audit heeft acht verplichte vergelijkingsfamilies en vijf
claim-eenheden tegen 34 primaire papers, officiële implementaties en beperkte
patentdatabasebronnen afgezet. De brede bouwstenen zijn bezet: alternatieve
routes, cachebewuste selectie, blockbrede expert-unies, tokenadaptieve
residual-bitplanes, neuronfijne sparsity en custom MoE-kernels/layouts.

CU1 (joint route+bits) en CU3 (randomized residual syndrome) krijgen uitsluitend
`possibly novel intersection`: de exacte doorsnede werd in de gerichte search
niet gevonden, wat geen bewijs van nieuwheid is. Beide mechanismen zijn in de
technische lijn bovendien gefalsificeerd. CU2 is `close/overlapping`; CU4 blijft
`not searched sufficiently` en is nooit dependency-vrij geïmplementeerd; CU5 is
`clearly prior art` en in CRAFT niet gebouwd.

De beperkte Google-Patents-pass vond onder meer residual quantization, volledige
plus partiële hot-expertbuffers en pseudo-random-projectioncompression. Dit was
geen claim chart of juridisch onderzoek en ondersteunt geen uitspraak over
patentability. Het deterministische JSON-resultaat bevat alle labels, bronnen,
zoekbeperkingen en technische statussen en sluit byte-exact met de generator;
vijf novelty-tests slagen.

Besluit: er is geen verdedigbare brede nieuwheidsclaim en geen Eureka. De
toelaatbare bijdrage is een preregistreerde negatieve-resultatenstudie met
exacte oracleplafonds en reproduceerbare falsificatie. Het masterverdict moet nu
de projectgates formeel sluiten; V4 Flash blijft buiten scope.

## 2026-08-10 — CRAFT masterverdict: gesloten zonder Eureka

De volledige geregistreerde hypothesefamilie sluit als `closed_no_eureka`.
Geen enkele kandidaat bewijst de zes conjunctieve revolutionaire V2-gates:
de score is 0/6 voor één en dezelfde kandidaat. Alle niet-geblokkeerde
hypothesen hebben een terminaal negatief besluit; H5, H9 en PACKED_RUNTIME
zijn volgens hun vooraf vastgelegde afhankelijkheden gestopt. Er is daarom
geen toegestane basis voor een packed-runtimeclaim, lange rollout of
tweede-model/V4-Flash-escalatie.

Mass-Budget delta=0,004 blijft als echte incrementele baseline overeind:
14,017% minder expertloads op het vaste WikiText-testvenster, KL 0,003704 en
relatieve CE -0,057% met 95%-blokinterval [-0,171%, +0,060%]. De bijbehorende
strict-I/O-boekhouding is 201,48/173,24 = 1,163011×, maar dit is uitdrukkelijk
geen gemeten latency-, throughput- of energieverbetering.

De sterkste lokale CRAFT-signalen vormen geen stapelbare doorbraak. H1 vraagt
3,12240 effectieve bits op laag 26 maar 3,70508 bits in de eerdere-laag-test.
Een ideale H3-policy met 25% BF16-atomen kost al 4,0 effectieve bits—gelijk
aan int4 vóór metadata—en faalt gelijktijdig full-depth. De 10%-variant is
theoretisch 1,6 bits, maar faalt dezelfde full-depthkwalificatie. H2 bereikt
slechts een unionfactor 0,7976 (ideaal 1,2538×). Deze factoren horen bij
verschillende interventies en mogen niet worden vermenigvuldigd.

De onafhankelijke bewijslaag sluit eveneens: de reproduceerbaarheidsaudit
heeft nul verplichte fouten en drie expliciet claimbegrenzende waarschuwingen;
de novelty-audit vindt geen verdedigbare brede nieuwheidsclaim. Het
byte-exacte masterresultaat heeft SHA-256
`6935f7c10f547d2c411c7ff6f370cea48552f8d2b83527284d4b12155ae00847` en het
leesbare masterrapport
`446eaf390e5e760100f6698c3d4761d9e2f7eb3a09651ff594fb4808b7e940a1`.

Besluit: het CRAFT-pakket wordt bevroren. Een vervolg is alleen geldig als
nieuw registry-item met een mechanistisch onafhankelijke hypothese en vooraf
vastgelegde oracle-, full-depth-, runtime- en tweede-modelgates. Er wordt niet
verder post-hoc getuned en DeepSeek V4 Flash wordt vanuit deze lijn niet
gedownload of getest.
