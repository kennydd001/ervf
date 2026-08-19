# S100 Phase 14 — volledig herzien onderzoeksrapport

**Datum:** 2026-08-19  
**Doelmodel:** exact `models/nemotron_3_5_lightning`  
**Parent:** huidige quality-green QFAST + `alpha=0.0003` ERVF-runtime  
**Hardwaregrens:** 8 GiB VRAM, maximaal 32 GiB RAM  
**Prestatieclaim:** interactieve single-stream decode; S100 betekent minstens 100 outputtokens/s

## Uitvoerend besluit

DFlash2 is inhoudelijk relevant en wordt toegevoegd als **Phase 14F**, maar niet
als kant-en-klare runtimepatch. Het is een model-specifieke, getrainde
speculative drafter met extra checkpointtensors en verifierlogica. Voor
Nemotron bestaat geen compatibele DFlash2-drafter. Een echte port vereist dus:

1. een Nemotron-specifieke draftconfig en engine-adapter die hidden states uit
   geselecteerde hybride targetlagen levert; de draftbackbone kan nog altijd een
   kleine dense Transformer zijn en hoeft het target niet architecturaal te
   kopiëren;
2. training op target hidden states en target outputdistributies;
3. een block verifier die meerdere tokens veel goedkoper verifieert dan B
   opeenvolgende decodepasses;
4. resident geheugen voor draftgewichten, draft-KV en selectorworkspace;
5. lossless greedy- en samplingverificatie.

Het huidige verifierpad vormt de hardste blokkade. Phase 12A mat een bitexacte
perfect-draft verifier van ongeveer 35,6 / 71,0 / 143,9 ms bij B=2/4/8. Zelfs
met 100% acceptatie en een drafter die letterlijk 0 ms kost, is het plafond
ongeveer 56 tokens/s. Phase 12C projecteerde B=4 na zijn gemeten componenttests
op ongeveer 60,9 ms, of 65,6 tokens/s bij perfecte acceptatie. DFlash2 kan de
acceptatielengte verhogen, maar kan deze verifierkosten niet wegtoveren.

Daarom is het correcte besluit:

- **ja, toevoegen** als een orthogonale train-or-kill onderzoekslijn;
- **nee, nog geen volledige drafter trainen** zolang de verifier geen positieve
  S100-budgetruimte heeft;
- wel nu meten of de twee DFlash2-ideeën overdraagbaar signaal vertonen op echte
  Nemotron-trajecten, zodat een latere verifierdoorbraak onmiddellijk kan worden
  benut.

## Wat DFlash2 werkelijk toevoegt

DFlash voorspelt zeven toekomstige tokens parallel in een blok van acht: één
reeds door het target geproduceerde anchor plus zeven draftposities. DFlash2
voegt twee learned componenten toe.

### 1. Suffix-decay correction

Een gewone parallelle drafter degradeert doorgaans naar het einde van het blok,
omdat latere posities geen betrouwbare informatie van eerdere voorstellen
krijgen. DFlash2 plaatst voor en na attention en MLP een grouped dynamic
depthwise convolution met twee causale taps:

```text
out[i,c] = sum_t (base[t,c] + delta[i,t,g(c)]) * x[i-t,c]
```

De dynamische coefficienten worden uit de huidige hidden state voorspeld en per
kanaalgroep gedeeld. Posities blijven parallel berekend, maar krijgen in een
lokale tweede stap informatie van de voorgaande positie.

### 2. Parallel path selection

DFlash v1 neemt per positie onafhankelijk de hoogste draftlogit. DFlash2 houdt
de top-16 kandidaten per positie bij en scoort overgangen tussen aangrenzende
posities:

```text
edge(previous -> candidate)
  = <A[previous] * project(hidden), B[candidate]> + unary[candidate]
```

Een dynamisch-programmeerwalk kiest daarna één coherent pad door de kandidaat-
lattice. Bij sampling wordt ook de werkelijke proposalverdeling `q` aan de
verifier doorgegeven, zodat maximal-coupling verification de targetdistributie
behoudt.

Dit zijn geen generieke postprocessoren. De convolutionele parameters,
codebooks, hidden projection en draftbackbone zijn getrainde checkpointgewichten.
De huidige publieke DFlash-repository bevat inferentiecode en checkpoints,
maar geen trainingsrecept of training-entrypoint. Een Nemotron-port kan dus niet
simpelweg een officieel trainingsscript hergebruiken.

## Externe evidence en overdraagbaarheid

De publieke Qwen3.8-27B DFlash2-config gebruikt vijf draftlagen, hidden size
5120, intermediate size 17408, block size 8, conv kernel 2, conv group size 16,
selector rank 256 en top-K 16. De drafter telt ongeveer 1,924 miljard parameters.
Het target embedding en de target LM-head worden gedeeld en zitten niet opnieuw
in de drafter.

Op één H200 rapporteert de SGLang-integratie bij concurrency 1 op GSM8K 236,1
tokens/s tegenover 68,9 autoregressief, met gemiddelde acceptatielengte 5,46.
Dezelfde publicatie rapporteert 2,67–3,43x op vijf workloads. De extra
convolution en selector kosten daar ongeveer 1,5–2,7% van de stap. De vLLM-PR
rapporteert vergelijkbare resultaten en meet bij batch 1 ongeveer 0,174 ms voor
conv plus selector tegenover een totale serving step van 20,70 ms.

Dat is sterke evidence dat de methode op een goed gebatchte GPU-verifier werkt,
maar geen bewijs voor onze omgeving:

- de H200 heeft een totaal andere geheugenbandbreedte, VRAM-capaciteit en
  matmul-economie;
- Qwen3.8 is een dense Transformer target, terwijl onze parent hybride
  Mamba/attention/MoE met gestreamde experts is; het DFlash-interface kan in
  principe nog werken, maar target-layer extraction en state/caching-semantiek
  moeten opnieuw worden ontworpen;
- de gepubliceerde target verifier verwerkt het blok in efficiënte matrix-
  kernels; onze exacte Phase-12-verifier unrolde grotendeels M=1-werk;
- de Qwen-drafter in BF16 gebruikt ongeveer 3,8 GB alleen aan gewichten;
- onze volledige targetruntime moet binnen 8 GiB blijven.

De llama.cpp-implementatie is een nuttige consumentenharde sanity check. Op een
M5 Pro 64 GB met Qwen3.8-27B Q4_K_M en B=8 rapporteert de open PR ongeveer
1,77–1,85x speedup en acceptatielengte rond 5. Daaruit volgt dat een gequantiseerde
drafter mogelijk is, maar ook dat de H200-factor niet universeel is.

Op 19 augustus 2026 is de DFlash2-integratie in SGLang gemerged; de vLLM- en
llama.cpp-integraties staan nog als open PR. Daarnaast documenteert een open
vLLM-issue dat het huidige niet-causale DFlash-attentionpad niet combineert met
FP8/quantized draft-KV. Daarom geldt FP8 draft-KV in onze geheugenmatrix
expliciet als hypothetisch; de harde resident-gate gebruikt BF16 draft-KV.

### Niet verwarren met naïeve multi-pass refinement

Een losse `hxri/dflash2`-fork voegde een tweede volledige draftpass toe. Die
variant zakte in de meegeleverde meting van DFlash 362,54 tokens/s en acceptatie
6,48 naar 59,94 tokens/s en acceptatie 1,19. Dat is niet Inco's DFlash2. Deze
bundel voegt daarom geen tweede draftpass toe; hij test specifiek de lokale
suffixcorrectie en candidate-pathselectie.

## Harde rekenvoorwaarde

Voor een verifiercyclus met tijd `T_verify`, draft/selector-overhead `T_draft`
en gemiddelde geaccepteerde outputlengte `A` geldt:

```text
throughput = 1000 * A / (T_verify + T_draft)
```

Voor S100 moet dus gelden:

```text
T_draft <= 10 * A - T_verify
```

De meest optimistische grens gebruikt `A=B` en `T_draft=0`. Als zelfs dan
`T_verify > 10*B`, is DFlash2 op dat verifierpad mathematisch uitgesloten voor
S100. Dat is momenteel het geval voor alle gemeten Phase-12A-blokken.

Deze stopregel voorkomt een dure training waarvan het absolute perfecte-draft
plafond al onder het doel ligt. De finale summary combineert bovendien de
werkelijk gemeten B=8 verifierkost met de onafhankelijke, selectorproxy- en
oracleacceptatielengtes uit 14F1. Dat blijft een zero-cost-drafter upper bound,
maar maakt direct zichtbaar hoeveel verifierbudget nog ontbreekt.

## Geheugenmodel voor een Nemotron-drafter

Phase 14F schaalt de openbare Qwen-DFlash2-vorm naar de werkelijke runtime-
dimensies. Voor iedere combinatie van 2–5 draftlagen en MLP-ratio 2,0 / 3,0 /
3,75 wordt berekend:

- attention-, MLP-, norm- en contextprojectieparameters;
- exact volgens de publieke referentie: conv kernel 2, group size 16,
  selector rank 256 en top-K 16;
- BF16-, FP8- en NVFP4-weightbytes;
- 12% workspace en minimaal 512 MiB veiligheidsreserve;
- draft-KV voor 4K, 32K en 128K context, zowel gegarandeerd BF16 als een
  expliciet hypothetische FP8-route.

Een Qwen-achtige vijflaagse variant op hidden 2688 wordt, inclusief een
conservatieve 8%-kalibratiefactor tegen het publieke parametercount, voorlopig
rond 0,70–0,75 miljard parameters geschat, afhankelijk van de werkelijke
attention Q/K/V-dimensies die de runtime inleest. Dat is ongeveer 1,30–1,40 GiB
in BF16 of 0,37–0,40 GiB aan NVFP4-code plus groupschalen, vóór workspace en
KV. De daadwerkelijke vrije
VRAM wordt gemeten nadat de huidige quality parent volledig is gebouwd. Geen
papieren geheugenwinst opent de gate als het resident gemeten budget ontbreekt.

## Volledig herzien Phase-14-plan

### Gesloten zonder rerun

**13A entropy census** blijft gesloten voor de geteste lossless
compressiehypothese.  
**13C temporal delta** blijft gesloten voor de geteste 99%-energiegate.

### 14D — native BF16 Tensor-Core survivor

Doel: vaststellen of de 5,47x B=4 componentceiling modelniveaukwaliteit
behoudt en voldoende breed geldt.

- alle live BF16-projecties;
- B=2/4/8, real-weightrotatie >4x L2;
- B=4 nuttige-rij-speedup >=2,5x;
- strict validation op `_02`;
- heldout `_03/_04` uitsluitend na groene strict validation;
- officiële CE/KL/top-1/top-5/domein/determinisme/finite-gates.

Uitgangsvlag:

```text
NATIVE_BLOCK_RUNTIME_BUILD_OPEN
```

### 14B2 — output-aware activation subspace

Doel: de onvolledige input-PCA van 13B vervangen door echte laagoutputfout.

```text
Y ~= X T C
```

- echte calibration- en validation-X/Y-paren;
- representatieve Mamba- en alle attentionprojecties;
- rank 32/64/128/192/256/384;
- BF16-afgeronde factoren;
- >=35% fysieke bytebesparing;
- output-NRMSE <=0,03;
- mean cosine >=0,9995;
- p95 relatieve rijfout <=0,08;
- minstens 80% van een familie moet slagen.

Uitgangsvlag:

```text
SUBSPACE_RUNTIME_BUILD_OPEN
```

### 14E2 — decoded expert shared basis

Doel: de ruwe-code-SVD van 13E vervangen door een realistische
activation-weighted outputtest.

- werkelijk gedecodeerde NVFP4 routed-upgewichten;
- vroege/middelste/late MoE-lagen;
- expert-axis ranks 4/8/16/32;
- gedeelde basis opnieuw naar NVFP4 CEIL;
- expert-specifieke residualblokken 6,25 / 12,5 / 25%;
- byte ratio <=0,70;
- validation sampled-GEMV NRMSE <=0,05;
- cosine >=0,999.

Uitgangsvlag:

```text
EXPERT_BASIS_RUNTIME_BUILD_OPEN
```

### 14F0 — DFlash2 economics en resident memory

Doel: vóór training bewijzen dat er theoretisch een positieve S100-budgetruimte
en resident geheugen bestaat.

De test leest Phase-12A, Phase-12C en iedere latere gemeten full-block verifier.
Per B en iedere acceptatielengte wordt berekend:

- perfecte-draft tok/s-ceiling;
- maximaal draft+selectorbudget voor S100;
- maximaal budget om de huidige autoregressieve parent te verslaan;
- gemeten versus geprojecteerde provenance.

Uitgangsvlaggen:

```text
DFLASH2_CURRENT_VERIFIER_S100_OPEN
DFLASH2_RESIDENT_MEMORY_OPEN_4K
```

Een geprojecteerde componentceiling mag de gemeten full-verifiergate niet
vervangen.

### 14F1 — Nemotron DFlash2 transfer proxy

Doel: zonder honderden GPU-uren training bepalen of de twee learned ideeën op
echte Nemotron-trajecten überhaupt voldoende signaal hebben.

#### Capture

- dezelfde 10 `_01` calibrationprompts en 10 `_02` validationprompts;
- 72 greedy targettokens per prompt;
- echte final-normalized hidden states;
- 32 deterministische anchorwindows per prompt;
- blok 8: één anchor plus zeven future slots.

#### Parallel base proxy

Een calibration-only low-rank regressie voorspelt uit alleen de final anchor
hidden zeven future hidden states tegelijk. Dit is bewust zwakker dan een echte
DFlash-drafter, die meerdere geselecteerde targetlagen ontvangt; een negatieve
proxy is daarom een goedkope screen closure en geen algemeen no-go-theorema.
Rank 64/128/192 wordt alleen op een interne calibrationsplit geselecteerd.

#### Dynamic two-tap suffix proxy

Voor group size 64/128 wordt per slot en groep een anchor-conditioned
dynamische coefficient geleerd. Alle basisposities worden eerst parallel
berekend; daarna wordt één shifted predecessor-tap toegepast. Dit is een
bewuste, goedkopere transferproxy voor het DFlash2-mechanisme, niet een claim
dat het de vijflaagse drafter vervangt.

Gate:

- minstens 10% lagere last-three hidden NRMSE, **of** minimaal +2 procentpunt
  top-16 recall op de laatste drie slots;
- slot 1 mag niet meer dan 1 procentpunt top-16 recall verliezen.

#### Candidate-lattice census

Voor basis en corrected states wordt de echte NVFP4 LM-head uitgevoerd en de
top-16 per slot bewaard.

Gemeten worden:

- onafhankelijke top-1 acceptatielengte;
- oracle-lattice acceptatielengte: het correcte token zit in top-16;
- selectorheadroom tussen beide;
- full-block coverage.

Gate:

- oraclegemiddelde inclusief anchor >=3,0;
- minimaal 0,75 token selectorheadroom.

#### Beperkte selectorproxy

Een vaste embedding-factorisatie projecteert predecessor, candidate en hidden
naar rank 32. Alleen één transition weight wordt op calibration gekozen; de
validatie gebruikt dynamic programming. Dit is een lower-capacity sanity check.
Mislukken sluit DFlash2 niet, maar toont hoeveel van de oracleheadroom een zeer
kleine selector kan benutten.

Uitgangsvlag:

```text
DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN
```

### Gecombineerde train-gate

```text
DFLASH2_TRAINING_BUILD_OPEN
  = current verifier S100-open
    AND resident 4K memory open
    AND Nemotron transfer signal open
```

Deze vlag geeft toestemming voor een afzonderlijke draft-trainingfase. Hij
claimt geen runtime, geen speedup en geen S100.

## Tri-state semantiek

Iedere finale vlag is:

- `true`: preregistered evidence groen;
- `false`: technisch volledig gemeten en de betreffende gate gefaald;
- `null`: evidence ontbreekt of de test faalde technisch.

`null` wordt nooit als wetenschappelijke falsificatie behandeld.

## Verwachte beslisboom

1. **Current verifier false**  
   Geen draftertraining. Eerst native/multi-row block verification breken.
2. **Verifier true, memory false**  
   Alleen verder met een aantoonbaar resident NVFP4/FP8 draftontwerp of minder
   cache; geen CPU-streamed drafter, omdat dat de draftlatency waarschijnlijk
   vernietigt.
3. **Verifier en memory true, transfer false**  
   Geen vijflaagse training. Eerst een kleiner MTP/DFlash distillation pilot om
   te controleren of de proxy te zwak was.
4. **Alle drie true**  
   Nieuwe, afzonderlijk preregistreerde Phase 15: datasetbouw, target hidden
   capture, draft training, quantisatie, lossless verifier en end-to-end
   wall-time.

## Wat deze bundel nadrukkelijk niet claimt

- De Qwen3.8-drafter werkt niet op Nemotron.
- Een Qwen/H200-speedup is niet overdraagbaar als percentage.
- Een oracle candidate lattice is geen implementeerbare selector.
- De 14F proxy vervangt geen drafttraining.
- Componenttimings vervangen geen volledige speculative cycle.
- Geen enkel Phase-14-resultaat is op zichzelf een S100-resultaat.

## Bronnen

- Inco AI, **DFlash2 announcement**: https://inco.ai/blog/dflash2/
- DFlash paper: https://arxiv.org/abs/2602.06036
- DFlash repository: https://github.com/z-lab/dflash
- Qwen3.8-27B DFlash2 checkpoint: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
- SGLang DFlash2 integration, merged PR #35371:
  https://github.com/sgl-project/sglang/pull/35371
- vLLM DFlash2 integration, PR #52816:
  https://github.com/vllm-project/vllm/pull/52816
- llama.cpp DFlash2 integration, PR #27342:
  https://github.com/ggml-org/llama.cpp/pull/27342
