# Pad naar 100 tok/s — eerlijke plafondanalyse en concrete routekaart

Datum: 2026-08-16 · status: synthese van deze sessie se volledige onderzoek,
geen nieuwe meting · doel: één plek die het doel (100 tok/s) confronteert met
alle fysieke feiten die deze sessie verzameld heeft, en een concrete,
afgebakende routekaart geeft voor wie dit oppakt.

## De twee routes, en waarom ze apart beoordeeld moeten worden

"100 tok/s" is nooit gespecificeerd als per-sequentie of aggregate. Dat
onderscheid is precies waarom dit document in twee helften valt.

## Route 1 — single-stream (per-sequentie), batch=1

**Status: fysiek zeer onwaarschijnlijk, mogelijk uitgesloten binnen deze
architectuur.**

- Roofline (geheugenbandbreedte-gebonden, hardware-eigenschap, niet
  modelafhankelijk): **165 tok/s bij ctx0**, gemeten via streaming-
  leesbandbreedte 338,4 GB/s (N5, `N1_N5_OWN_HYPOTHESES_REPORT_2026-08-15.md`).
- Huidig record (V6, alle batch=1-hefbomen: graph-residentie + selectieve
  ERVF + gebatchte down_proj/up_proj/accumulate-kernels + per-laag
  capaciteitstuning): **47,41 tok/s = 28,9% van de roofline.**
- 100 tok/s zou **60,6% van de roofline** vragen — meer dan het dubbele van
  wat de meest geoptimaliseerde stack tot nu toe haalt, ná uitputting van
  elke binnen-batch=1-hefboom die deze sessie kon vinden (down_proj/up_proj/
  accumulate/panel_scan/reduce_partials allemaal gebatcht; capaciteitstuning
  bevestigd near-optimal via sweep; gather/down_masked-batching geprobeerd en
  afgewezen op VRAM).
- GPU-klokverval (36% SM-klokdaling onder aanhoudende belasting,
  `diag_lmhead_throttle_check.py`) bedreigt de roofline zelf **niet**
  (`clocks.mem` blijft vlak) — dus dit is geen extra onbenutte hoofdruimte.
- **Conclusie**: zonder een architecturaal andere aanpak (niet binnen
  batch=1) is 100 tok/s single-stream niet aannemelijk bereikbaar op deze
  hardware met dit model. Dit is geen educated guess — het is een
  bandbreedte-plafond, en 47,41 zit al op bijna 29% ervan na uitgebreide
  optimalisatie.

## Route 2 — aggregate doorvoer, batch>1

**Status: theoretisch plausibeler, praktisch nog niet gedemonstreerd binnen
deze sessie — het enige pad dat 100 niet a priori uitsluit.**

### Wat fysiek bewezen is (bitexact, dus vertrouwbaar)

1. Het kernmechanisme (expert-fetch delen over sequenties) is **correct en
   sneller in isolatie**: up_proj 1,71-2,89×, down_proj 1,91×, gecombineerd
   op één laag 1,209× (+20,9%).
2. Houdt stand onder realistische voorwaarden: warme/evoluerende cache
   (27,6% minder missers, niet cold-cache-only), staggered posities
   (continuous batching verzwakt de unie slechts licht, 89,4%→91,4%).
3. VRAM is **geen** blokkade buiten graph-capture om (60,16 MiB/sequentie,
   ruimte voor N tot 30 bij het eager-bedrijfspunt).
4. **Eerste echte end-to-end meting** (volledig 52-lagen model, meerdere
   echte stappen, N=2, bitexact tegen onafhankelijke `_moe_dev`-referentie):
   naive (alleen incidenteel warm-cache-hergebruik) geeft al een reële
   **+5,4% aggregate** winst zonder enige nieuwe code in de hete lus.
5. De EXPLICIETE unie-gevoede deling, volledig geïntegreerd in de echte
   staplus en met de al bestaande gebatchte V5/V6-kernels versneld, is
   **bitexact correct** maar haalt in deze sessie se Python-georkestreerde
   vorm slechts **11,23 tok/s aggregate** (nog onder de naive baseline van
   31,41 en onder solo 29,80) — 4,19× sneller dan de eerste werkende versie,
   maar het PCIe-bandbreedte-gebonden deel (gather + up_proj-fetch, ~49%
   van de tijd) is dicht bij zijn fysieke vloer voor verdere
   launch-batching.

### Waarom dit nog niet bij 100 in de buurt komt, en wat daar wél voor nodig is

De **grove, expliciet-gelabelde bovengrens** uit
`BATCH_ARCHITECTURE_DESIGN.md` (aanname van perfecte MoE-deling, rest
ongewijzigd) kwam op **~114 tok/s aggregate** — een rekensom, geen meting,
en inmiddels tweemaal naar beneden bijgesteld door echte metingen:

- Mamba wordt **duurder** per sequentie bij grotere N (~15% straf, N=8-16).
- lm_head, nooit eerder genoemd als risico, blijkt een **grotere** straf te
  hebben (~19-24%) op de duurste GEMV van het model.
- Beide zijn nu verklaard als reëel GPU-SM-klokverval (36%) onder
  aanhoudende belasting — een fysiek, gemeten fenomeen, geen ruis.

Een realistische bovengrens ligt dus **onder** 114, misschien rond
90-105 tok/s aggregate bij perfecte engineering — dicht genoeg bij 100 om
het de moeite waard te maken, maar met een reële kans om er net onder te
blijven zelfs bij perfecte uitvoering.

**Wat "perfecte engineering" concreet zou vereisen (niet gedaan deze
sessie, wel nu precies afgebakend):**

1. **Device-only routing-unie-berekening.** Deze sessie se
   `shared_moe_layer`-integratie leest routing-ids terug naar de host
   (`cp.asnumpy`) om de Python-unie te berekenen — een host-sync per
   sequentie per laag. Een echte implementatie zou de unie **op het device**
   moeten berekenen (een kleine nieuwe kernel: N×top_k ids in, gededupliceerde
   lijst + telling uit, zonder host-round-trip).
2. **Eén CUDA-graph voor de hele multi-sequentie-staplus**, met een
   actief-masker voor continuous batching (zoals `BATCH_ARCHITECTURE_DESIGN.md`
   stap 8 al aangaf) — vangt de PCIe-fetch en het rekenwerk in dezelfde
   graph-residentiewinst die V4-V6 al voor batch=1 bewees (+22-33%).
3. **Stream-overlap tussen PCIe-transfer en rekenwerk** — production
   `_moe_dev` doet dit al voor shared-expert-vs-up_proj-fetch (één specifiek
   geval); een volledige batch>1-integratie zou dit moeten uitbreiden naar
   gather-vs-ander-rekenwerk. Dit is de hefboom die deze sessie identificeerde
   maar niet bouwde na de kernel-batching-ronde (die loste
   launch-overhead op, niet PCIe-overlap).
4. **N_max-vaste-groottebuffers voor continuous batching**, niet de
   Python-dict-per-stap-aanpak van deze sessie se prototypes.

Punten 1-3 zijn stuk voor stuk **kleiner dan een volledige runtime-herbouw**
— elk is een week-schaal CUDA-taak, niet een sessie-taak, maar ook niet de
ongedefinieerde "meerdere weken" die het eerdere ontwerpdocument noemde
zonder specifieke stappen. Met deze sessie se werk (bitexacte
kernels al gebouwd, correctheidsmechanisme al geverifieerd, exacte
knelpunten al geprofileerd) is de resterende afstand tot een eerste echte
geëngineerde poging aanzienlijk kleiner dan aan het begin van de dag.

## Eerlijke eindconclusie

- **Single-stream 100 tok/s: fysiek zeer onwaarschijnlijk** op deze
  hardware/model-combinatie. Het bandbreedteplafond (165) en het huidige
  record (47,41, 28,9%) laten weinig ruimte, en elke binnen-batch=1-hefboom
  die deze sessie kon vinden is al uitgeput of afgewezen.
- **Aggregate 100 tok/s via batch>1: het enige pad dat niet a priori
  uitgesloten is**, maar vraagt een echte (niet-Python-georkestreerde)
  runtime-integratie die deze sessie bewust niet heeft geprobeerd te
  forceren — wel heeft deze sessie het mechanisme bitexact bewezen, de
  risico's stuk voor stuk gemeten (niet aangenomen), en de resterende
  technische stappen (device-only unie-kernel, graph-residentie,
  stream-overlap) precies benoemd in plaats van vaag "meer engineering
  nodig" te laten staan.
- **Wat NIET gedaan is, met opzet**: de volledige multi-week CUDA-
  herimplementatie zelf. Dat blijft een bewuste keuze — niet omdat het
  onmogelijk is, maar omdat het project se eigen methodologie ("isoleer,
  verifieer, integreer pas als het klopt") een rushed, halfgeverifieerde
  poging in één sessie afraadt, en elke stap tot nu toe die discipline heeft
  gevolgd (inclusief het corrigeren van eigen fouten wanneer gevonden — de
  race-condition-misdiagnose, de cache-scan-verklaring, de pos-bug — allemaal
  vóór ze een verkeerde conclusie hadden kunnen laten staan).

## Voor wie dit oppakt: prioriteitsvolgorde

1. Device-only unie-routing-kernel (sluit de grootste resterende
   Python-host-sync-kloof).
2. Stream-overlap voor gather (PCIe) vs. ander rekenwerk binnen dezelfde
   laag.
3. CUDA-graph-residentie voor de multi-sequentie-staplus (grootste
   enkelvoudige hefboom, gezien V4-V6's eigen +22-33%-precedent voor
   batch=1).
4. Pas dan: een eerlijke, opnieuw-gemeten aggregate tok/s-claim — met
   dezelfde bitexacte-correctheidspoort-discipline als elke stap hiervoor.
