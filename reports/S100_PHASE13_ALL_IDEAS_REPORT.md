# S100 Phase 13 — volledig ideeënrapport

Datum: 2026-08-18  
Checkpoint: `models/nemotron_3_5_lightning`  
Branch: `agent/s100-phase13-subspace-entropy`

## Samenvatting

De vijf oorspronkelijke discovery-sporen zijn één voor één uitgevoerd. De
aanvullende ideeën uit de nieuwe hoofdgedachte zijn daarna als afzonderlijke
component- of falsificatiescreens getest. Geen enkel resultaat opent op dit
moment een modelpromotie: er is nog geen end-to-end heldout quality gate groen
voor een gewijzigde runtime.

De sterkste positieve signalen zijn:

- native BF16 matrixblokken zijn lokaal veel sneller dan de huidige row-wise
  GEMV-kernel;
- een logit-margin correleert met stabiliteit onder gecontroleerde verstoring;
- een exact top-K witness kan een onnauwkeurige shortlist veilig her-ranken in
  de gecontroleerde logit-screen.

De belangrijkste negatieve signalen zijn:

- de eenvoudige Subspace-Residual-route heeft hoge echte outputfout en een
  zeer kleine validation-fast-path;
- de geteste lokale palette-entropycodec vergroot de data in plaats van die te
  verkleinen;
- een gedeelde expertbasis, zoals hier gemeten op de FP8-codevlakte, geeft
  onvoldoende reconstructiekwaliteit bij een nuttige bytewinst;
- de veilige residual-gated pipeline is zonder een echte fused/persistent
  kernel trager dan exact rekenen.

## Testmatrix

| Idee | Test | Resultaat | Status |
|---|---|---|---|
| Lossless entropy census | 13A | Mamba-FP8 best bleef 8 bits/weight; resident palette circa 96,8% van raw bytes | Gesloten als eenvoudige statistische route |
| Activation-subspace census | 13B | Activatieresidu gemeten op calibration/validation; nog geen outputbewijs | Discovery, niet promotie |
| Temporal delta census | 13C | Top-256 delta-energie bleef ver onder de 99%-gate; MoE circa 45% | Gesloten voor simpele coordinate-top-k-route |
| Native BF16 Mamba block | 13D | B=4 mediaan circa 5,47x sneller; row-argmax circa 97,9% | Positief componentresultaat |
| Expert shared basis | 13E | Rank 32: circa 0,40–0,41 NRMSE bij ideale circa 44,4% bytewinst | Niet promoten |
| Volledige Subspace-Residual `WU` | 13F | Rank 256 output-NRMSE circa 0,375 Mamba-in en 0,587 Mamba-out; 50%-gate gaf slechts circa 2,4%/2,3% validation-fast tokens | Simpele versie gefalsificeerd |
| Werkelijke lossless tilecodec | 13G | Exacte roundtrip, maar 4/5/6-bit palette gemiddeld 1,226x/1,282x/1,303x dataomvang; decode circa 1,4–1,7 MiB/s op CPU | Deze codec gesloten |
| Native BF16 attention / FP8 / NVFP4 | 13H | BF16 Q/O circa 6,24x op B=4; NVFP4 packed x2 niet native consumeerbaar; geen FP8 Mamba-case beschikbaar | BF16 positief; overige paden open |
| Decision-directed margin | 13I | Bij 10% logitruis: hoge-margin helft 100% stabiel, lage-margin helft 95,0% | Positief gate-signaal |
| Cross-layer activation basis | 13J | Rank 256 attention-residu 0,329 apart versus 0,429 gedeeld; MoE circa gelijk | Geen positief signaal |
| Exact top-K witness | 13K | K=16 behield de echte top-1 in alle samples bij 10% gecontroleerde logitruis | Positief, maar nog synthetische shortlist |
| Persistent compressed-layer engine | 13L | Subspace-only circa 6,96x, maar NRMSE circa 0,95; gated eager-pipeline 0,82x op B=4 | Control-flow prototype, geen engine |

## Wat precies bewezen is

### 1. Subspace-Residual ERVF

De centrale formule is daadwerkelijk op outputniveau getest:

`y_approx = (W U)(Uᵀ x)`

`W U` is op de GPU berekend met echte BF16 Mamba-gewichten. De residunorm is
op calibrationdata gekalibreerd en op aparte validationdata gebruikt voor een
exacte fallback-simulatie. De eenvoudige globale variant houdt echter te weinig
outputinformatie vast. De activatiebasis alleen is dus geen bewijs voor een
bruikbare 50%-fast-path.

Wat nog ontbreekt voor een eventuele nieuwe variant: per-layer bases met
geleerde outputfout, echte residualkolommen, Mamba state-refresh, en een
gewijzigde autoregressieve runtime met officiële quality gates.

### 2. Lossless entropy

De volledige entropy-census en een werkelijke exacte tile roundtrip zijn beide
gedaan. De onderzochte lokale palettecodec is niet voldoende: metadata en
escapes kosten meer dan de symbolen besparen. Dit sluit niet iedere ANS-,
dictionary- of hardwaredecoder uit, maar die varianten zijn nog niet gebouwd
of bewezen.

### 3. Native blokken

De oorspronkelijke bitexacte reductiebeperking is niet langer als eis gebruikt
in de native componenttests. BF16 Mamba en BF16 attention laten sterke lokale
speedups zien. Dat is echter geen 100 tok/s-resultaat: de tests meten losse
matrixblokken, niet een complete speculative prefill met state, routing,
heldout fidelity en fallback.

FP8 Tensor-Core Mamba kon op deze checkpoint niet als aparte native-case worden
gemeten omdat de geselecteerde Mamba-populatie geen `fp8_tensor`-matrix bood.
NVFP4 bleef packed (`float4_e2m1fn_x2`) en had geen werkende native PyTorch
matrixroute in de gebruikte stack.

### 4. Expertbasis

Alle 128 experts op drie representatieve MoE-lagen en beide projecties zijn
gescreend op de FP8-codevlakte. De theoretische opslagwinst is aanwezig, maar
de reconstructiefout blijft te hoog bij de ranks die nog bytewinst geven.
Decoded numerieke output, routed activation quality en een gedeelde-expertkernel
zijn nog niet bewezen.

### 5. Adaptive beslissingen

De decision-margin screen ondersteunt het idee dat moeilijke tokens vaker een
exacte fallback nodig hebben. De top-K witness screen ondersteunt een tweede
veiligheidsmechanisme: een fast shortlist kan exact worden gecertificeerd door
alleen de kandidaatrows opnieuw te berekenen.

Beide zijn nog geen geïntegreerde runtime. De gebruikte perturbaties waren
gecontroleerd/synthetisch; ze vervangen geen echte approximate lm_head,
subspace- of entropy-score.

## Niet volledig getest of bewust nog open

Deze onderdelen uit de hoofdgedachte zijn nog niet als volwaardige runtime
bewezen:

- native FP8 Tensor-Core uitvoering op een echte FP8 Mamba-case;
- native NVFP4 Tensor-Core uitvoering met correcte packed-scale semantics;
- cuBLASLt/CUTLASS-specifieke kernelselectie buiten de PyTorch/cuBLAS-route;
- Mamba forced-exact refresh om state drift te begrenzen;
- echte sparse residual-column kernels;
- een volledig persistent custom CUDA-kernelpad;
- een gecomprimeerde lm_head-score met echte exact top-K rerank;
- een volledig geïntegreerde architecturele combinatie van entropy,
  subspace, gate, fallback, routing en heldout generation;
- officiële quality/fidelity-gates voor ieder approximate pad.

Deze open punten zijn niet stilzwijgend als geslaagd beschouwd.

## Commitoverzicht

Elke test is afzonderlijk gecommit en naar dezelfde branch gepusht:

- [13A `61266fc`](https://github.com/kennydd001/ervf/commit/61266fc) — entropy census
- [13B `d8d8c9e`](https://github.com/kennydd001/ervf/commit/d8d8c9e) — activation subspace census
- [13C `bf5e95a`](https://github.com/kennydd001/ervf/commit/bf5e95a) — temporal delta census
- [13D `14e64df`](https://github.com/kennydd001/ervf/commit/14e64df) — native Tensor-Core BF16 block
- [13E `968d601`](https://github.com/kennydd001/ervf/commit/968d601) — expert shared basis
- [13F `bba2fd3`](https://github.com/kennydd001/ervf/commit/bba2fd3) — Subspace-Residual output test
- [13G `a93f0fd`](https://github.com/kennydd001/ervf/commit/a93f0fd) — lossless entropy codec
- [13H `e6e2b4b`](https://github.com/kennydd001/ervf/commit/e6e2b4b) — native datatype blocks
- [13I `1d4255b`](https://github.com/kennydd001/ervf/commit/1d4255b) — decision margin
- [13J `99cfdb5`](https://github.com/kennydd001/ervf/commit/99cfdb5) — cross-layer basis
- [13K `f63707a`](https://github.com/kennydd001/ervf/commit/f63707a) — exact top-K witness
- [13L `7c85a1c`](https://github.com/kennydd001/ervf/commit/7c85a1c) — compressed pipeline prototype

## Eindconclusie

De nieuwe hoofdgedachte is inhoudelijk serieus getest, maar de volledige
doorbraakclaim is niet bewezen. De meest kansrijke vervolgrichting is nu niet
een globale low-rank compressie. De evidence wijst eerder naar een combinatie
van:

1. native BF16 block execution waar de matrixvorm dit toelaat;
2. een decision-margin fallback;
3. exact top-K witnessing voor logits/routing;
4. alleen daarna een nieuwe, output-aware residualrepresentatie.

Op basis van deze metingen is er geen verantwoord bewijs voor een 100 tok/s
claim en is geen gewijzigde runtime naar productiepromotie geopend.
