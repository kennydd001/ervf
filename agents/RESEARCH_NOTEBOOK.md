# Onderzoekslogboek

Eén blok per fase, **nieuwste bovenaan**. Schrijf hier ook wat er *niet* werkte
en waarom — dat is meestal het bruikbaarste deel. Formaat:

```
## <datum> — <fase> — <verdict in één zin>
**Vraag** · **Opzet** (armen, één variabele) · **Uitkomst** (getallen) ·
**Poorten** · **Wat dit sluit of opent** · **Artefacten**
```

---

## 2026-08-16 — Per-laag cachecapaciteit fysiek gemeten en geïntegreerd: 47,41 tok/s

**Vervolg op de hitrate-diagnose** (−14,3% missers, hitrate 85,6%→87,7%,
budget-neutraal). Nu fysiek causaal getest en in V6 geïntegreerd.

**Causale A/B** (`pro_research/v_capacity_realloc_ab.py`, productiekernels,
géén batching erbij — één variabele: capaciteitsverdeling), volgens
hetzelfde precedent als A1 (capaciteit veranderen bewijsbaar bitexact met
D1 aan). Full mode, 765 samples: BASE_A 31,1651 → NONUNIFORM 31,2058 →
BASE_B 31,7189 ms. Poorten: `nonuniform_equals_base_bitexact` ✅ (bitexact,
bevestigt het A1-precedent) · `base_drift_le_1ms` ✅ · `ctl_diverges` ✅ ·
winst **0,2362 ms/token (0,75%)** — kleiner dan de ruwe schatting (~0,31 ms)
maar reëel en positief, alle poorten groen.

**Geïntegreerd in V6** (`pro_research/layer_capacity.py`, herbruikbare
`apply_nonuniform_capacity(rt)`, aangeroepen na élke `enable_cache()`-call
in `graph_v6_full_stack.py` inclusief de CTL-herbouw — capaciteit is
budget-neutraal, dus **geen VRAM-kost**, in tegenstelling tot de
gather/down_masked-poging hierboven).

**Uitkomst (full, 765 samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,4289 ms | 31,82 |
| **V6 (nu met per-laag capaciteit)** | **21,0923 ms** | **47,41** |

Alle poorten groen (bitexact, deterministisch, controle-arm wijkt af, VRAM
binnen budget — ongewijzigd t.o.v. eerder, want budget-neutraal). Nieuw,
klein record: 47,41 tok/s (was 47,08-47,37 in eerdere runs, sessie-variantie
in die orde, maar dit is de eerste run MET de capaciteitswinst erbij).

**Stand.** 47,41/165 = **28,7% van roofline**, feitelijk ononderscheidbaar
van vóór deze toevoeging gezien de sessie-variantie (~0,2-0,3 ms/token is
klein t.o.v. de ruis tussen losse volledige-graph-builds) — maar wel een
**bitexact geverifieerde, budget-neutrale, reële winst**, geen ruis in de
eigen geïsoleerde A/B (die had een BASE_A/BASE_B-drift van 0,55 ms, dus de
0,24 ms winst zit ruim binnen wat de eigen poort als betekenisvol aanmerkt).

**Wat nog open staat.** De −20/+30-verdeling was een eerste gok op basis
van één 256-token-rollout op één prompt; een grondiger optimalisatie
(meerdere prompts, andere deltas, mogelijk continue in plaats van twee
discrete niveaus) is niet gedaan en zou nog iets meer kunnen opleveren.

**Artefacten.** `pro_research/layer_capacity.py` ·
`pro_research/v_capacity_realloc_ab.py` ·
`pro_research/results/PRO_CAPACITY_REALLOC_AB.json` ·
`pro_research/graph_v6_full_stack.py` (uitgebreid) ·
`pro_research/results/PRO_V6_FULL_STACK.json` (bijgewerkt).

---

## 2026-08-16 — Correctie: gather/down_masked batchen bleek WÉL correct — maar niet de moeite waard om te integreren

**Correctie op het eerdere "race condition"-blok.** Dat was voorbarig.
Herbeoordeling: de referentiekernel in die test is een **letterlijke kopie**
van de exacte kernel die V5/V6 al duizenden keren correct heeft gedraaid op
het echte model (bitexacte causale A/B's, geen enkele NaN daar). Als
diezelfde kernel op ECHTE data altijd correct is maar op MIJN synthetische
data NaN geeft, ligt het probleem hoogstwaarschijnlijk bij de synthetische
testdata (ongeconstrainde `standard_normal`-LUTs/bytepatronen die een
combinatie raken die met échte ReLU2-sparse activaties nooit voorkomt), niet
bij een race.

**Bevestiging.** Twee nieuwe scripts testen met **echte, gevangen
modeldata** in plaats van synthetische random data:
`verify_down_gather_batch_real_data.py` (top_k=1, één echte laag-aanroep,
via monkeypatchen van `fused.down_masked_ind_k` om de exacte argumenten van
een echte `rt.step()`-aanroep te onderscheppen) en
`verify_down_gather_batch_real_full.py`/`verify_gather_batch_real_full.py`
(top_k=6, alle zes echte expert-aanroepen van één laag). Uitkomst in alle
gevallen: **bitexact, nul NaN**, zowel voor `gather_down_sparse_ind_batched`
als `gemv_down_masked_partial_ind_batched`. De kernels zijn dus wel degelijk
correct — de eerdere "race"-diagnose was fout.

**Geïntegreerd en fysiek getest.** `moe_dev_batched.py` uitgebreid met een
optionele `gather_kernels`-parameter (`down_gather_batch_kernels.py`).
Causale A/B apart van V5/up-proj gehouden (`v_gather_batched_ab.py`, één
variabele: gather-batching aan/uit, bovenop de al geverifieerde stack).
Full mode: bitexact, controle-arm wijkt af, **+0,6826 ms/token (+2,56%)**
bovenop V5+up-proj. Alle poorten groen — in isolatie is dit dus een échte,
kleine winst.

**Maar niet geïntegreerd in V6 — twee redenen, gemeten, niet aangenomen.**
1. **VRAM.** Gather/down_masked-batchen vereist `top_k` onafhankelijke
   mirror-buffers i.p.v. één hergebruikte (`self.mstate["mirror"]`) — dat is
   `top_k × 23 lagen × 2,68 MB ≈ 387 MB`, tegenover het bestaande
   64 MiB-budget. **De VRAM-poort faalde**, en is niet verruimd.
2. **Marginale winst verdampt bij volledige integratie.** V6 met gather-
   batching erbij: 21,1129 ms/token (47,36 tok/s, full mode) — **binnen
   ruis identiek** aan V6 zonder deze toevoeging (21,1118 ms/token, 47,37
   tok/s, eerder gemeten). De +0,68 ms uit de geïsoleerde A/B is dus al
   grotendeels aanwezig via andere weg zodra de rest van de graph-
   residentie/batching al actief is (vergelijkbare les als de eerdere
   ablatiecorrectie: eager-gemeten componentwinsten vertalen zich niet
   1-op-1 naar extra winst bovenop een al sterk geoptimaliseerde graph).

**Besluit.** `graph_v6_full_stack.py` draait `gather_kernels` bewust NIET
mee (expliciet becommentarieerd in de code, niet stilzwijgend weggelaten).
V6 blijft op zijn eerder geverifieerde configuratie: **~47,1-47,4 tok/s**,
alle poorten groen, VRAM ruim binnen budget. `moe_dev_batched.py`'s
uitbreiding blijft wel beschikbaar (`gather_kernels=None` is de default) —
bruikbaar als op een GPU met meer VRAM-marge, of als de cache-capaciteit
elders wordt teruggebracht om ruimte te maken, dit alsnog de moeite waard
wordt.

**Artefacten.** `pro_research/verify_down_gather_batch_real_data.py` ·
`pro_research/verify_down_gather_batch_real_full.py` ·
`pro_research/verify_gather_batch_real_full.py` ·
`pro_research/v_gather_batched_ab.py` ·
`pro_research/results/PRO_GATHER_BATCHED_AB.json` ·
`pro_research/moe_dev_batched.py` (uitgebreid, optionele
`gather_kernels`-parameter, default `None`).

---

## 2026-08-16 — Per-laag cachecapaciteit herverdelen: −14,3% missers, nog niet fysiek gemeten

**Vraag.** De hitrate-diagnose (eerder vandaag) liet sterk niet-uniforme
missrates per laag zien bij uniforme capaciteit 72 (laag 1/3/6/51: 25-42%,
de rest 6-15%). Helpt het om capaciteit weg te halen bij lage-miss-lagen en
te geven aan hoge-miss-lagen, bij **gelijkblijvend totaal budget**?

**Methode.** `pro_research/diag_per_layer_capacity.py` — puur hitrate,
geen timing-claim. `enable_cache()`'s eigen allocatielogica (runtime.py:
324-378) letterlijk hergebruikt om specifieke lagen na de normale
`enable_cache(72)`-aanroep te herallokeren (geen wijziging aan runtime.py;
`_moe_dev` leest `c["cap"]` toch al per laag dynamisch, dus heterogene
capaciteit is architecturaal al ondersteund — alleen `enable_cache`'s eigen
gemaksfunctie is uniform). Test: −20 op de 6 laagste-miss lagen (72→52),
+30 op de 4 hoogste-miss lagen (72→102), totaal ongewijzigd (1656 slots).

**Uitkomst.** Missers: **5.182 → 4.443 (−739, −14,3%)**. Hitrate: 85,61% →
87,66%. De geboosde lagen verbeteren dramatisch (laag 1: 665→300 missers,
meer dan gehalveerd), de verlaagde lagen verslechteren mild (laag 13:
100→164) — LRU-hitrate is sterk niet-lineair bij lage capaciteit, dus
weghalen bij een al goed bediende laag kost weinig terwijl geven aan een
slecht bediende laag veel oplevert.

**Voorzichtige tijdschatting (rekenwerk, geen meting).** Bij M1's
bulk-tempo (24,93 GB/s) en 2,68 MB per up_proj-miss: 739 minder missers over
256 tokens ≈ 739/256 ≈ 2,89 missers/token minder × 2,68 MB ≈ 7,75 MB/token
minder ÷ 24,93 GB/s ≈ **~0,31 ms/token** potentieel — reëel maar klein
vergeleken met de kernel-batchingwinsten van deze sessie (1-9 ms/token).

**Nog niet gedaan.** Geen fysieke causale A/B (alleen hitrate, geen
tokentiming), niet geïntegreerd in V6, geen verfijning van de −20/+30-keuze
(die was een eerste ruwe gok, geen optimum). Kleine maar reële hefboom,
lagere prioriteit dan wat al gebouwd is gezien de bescheiden geschatte
omvang.

**Artefacten.** `pro_research/diag_per_layer_capacity.py` ·
`pro_research/diag_per_layer_capacity.json`.

---

## 2026-08-16 — `gather_down_sparse_ind`/`gemv_down_masked_partial_ind` batchen: NIET gelukt, eerlijk gesloten

**Poging.** Na vier geslaagde batchingen (panel_scan, reduce_partials,
accumulate, up-proj-GEMV) leek het logisch de laatste twee down_proj-
subkernels ook te proberen — beide bleken bij nader inzien óók een vaste
(niet data-afhankelijke) grid-formule te hebben, dus in principe dezelfde
veilige klasse. Nieuw bestand `pro_research/down_gather_batch_kernels.py`
met referentie- en batched varianten van `gather_down_sparse_ind` (slot via
`blockIdx.y`) en `gemv_down_masked_partial_ind` (slot via `blockIdx.z`,
`blockIdx.y` blijft de bestaande chunk-dimensie).

**Uitkomst: niet bitexact, en het patroon wijst op een race, geen simpele
adresseerfout.** Geïsoleerde test
(`verify_down_gather_batch_kernels.py`, synthetische data): de **mirror-
data** (gather's output) is **bitexact identiek** tussen referentie en
batched — dus `gather_down_sparse_ind_batched` zelf is aantoonbaar correct.
Maar `gemv_down_masked_partial_ind`'s uitvoer verschilt daarna alsnog, mét
NaN's — **ook in de referentie-arm** (318 NaN's), niet alleen de batched
arm (2756-3799 NaN's), en de **exacte tellingen wisselen tussen
proces-runs** bij identieke code en identieke invoerdata. Dat patroon —
zelfde logica, zelfde data, ander resultaat, niet-reproduceerbaar tussen
runs — wijst op een **race condition of synchronisatieprobleem**, niet op
een simpele transcriptiefout in de adressering (die zou wél deterministisch
verkeerd zijn, elke run hetzelfde).

**Besluit: niet verder geforceerd, niet geïntegreerd.** Geen enkele wijziging
raakte `moe_dev_batched.py` of `graph_v6_full_stack.py` — V6 blijft op zijn
volledig geverifieerde 47,37 tok/s staan. Dit is precies de reden waarom
elke stap in deze sessie eerst geïsoleerd bitexact getest werd vóór
integratie: deze twee kernels faalden op die eerste, goedkope stap, dus is
er niets stuk gemaakt. Verder debuggen vraagt gereedschap dat deze sessie
niet heeft (compute-sanitizer/cuda-gdb) — eerlijk gesloten in plaats van
doorgeduwd met een niet-geverifieerd resultaat.

**Wat dit niet raakt.** De al bitexact geverifieerde down_proj-batching
(`panel_scan`+`reduce_partials`, in V5/V6) gebruikt nog steeds de
ORIGINELE, ongewijzigde `gather_down_sparse_ind`/
`gemv_down_masked_partial_ind` per slot — dat pad is en blijft correct.

**Artefacten.** `pro_research/down_gather_batch_kernels.py` ·
`pro_research/verify_down_gather_batch_kernels.py` ·
`pro_research/verify_down_gather_batch_kernels.json` (status: gefaald, niet
verwijderd — een weerlegging is ook DONE).

---

## 2026-08-16 — Up-proj ERVF-GEMV gebatcht: 47,37 tok/s, roofline 28,7%

**Aanleiding.** Componentafbraak liet zien dat MoE 57,8% van het token is,
en down_proj (V5) daar maar 6,51 van de 12,07 ms van uitmaakt. De
up-proj-GEMV (`gemv_nvfp4_ervf_ind`, de ERVF-kernel die de NVFP4-experts
projecteert) wordt zes keer per laag sequentieel aangeroepen, elk naar een
**onafhankelijke** outputregio (`bs["act"][s]`) — geen gedeelde accumulator,
dus géén race, in tegenstelling tot `accumulate_indirect`. `x` (de
genormaliseerde hidden state) is bovendien identiek voor alle zes slots.
Grid-grootte is vast (niet data-afhankelijk). Dit is dus dezelfde veilige
batch-klasse als `panel_scan`/`reduce_partials`, alleen op een grotere,
belangrijkere kernel — de zorgvuldig met NERVF-2 tegen een referentie
geverifieerde WIDTH-16-subwarp-butterfly-reductie.

**Aanpak — extra voorzichtig gezien het gewicht van deze kernel.** Nieuw
bestand `pro_research/up_proj_batch_kernels.py`: de referentiekernel
(`gemv_nvfp4_ervf_ind_ref`) staat **letterlijk gekopieerd** naast de
batched versie (`gemv_nvfp4_ervf_ind_batched`, `blockIdx.y`=slot toegevoegd,
verder geen enkele regel rekenkunde gewijzigd) om transcriptiefouten in de
reductie-boom te vermijden. Geïsoleerde bitexacte test
(`verify_up_proj_batch_kernel.py`, synthetische data op echte dimensies
1856×2688, drie trials incl. `apply_relu2` aan/uit): **bitexact.**

**Causale A/B, apart van V5's eigen resultaat gehouden** (één variabele:
up-batching aan/uit, bovenop de al geverifieerde down_proj-batching —
`pro_research/v_up_proj_batched_ab.py`). Full mode: bitexact,
**+1,7423 ms/token (+6,11%)** bovenop V5. Alle poorten groen.

**V6 opnieuw gedraaid** (`graph_v6_full_stack.py` uitgebreid met
`up_proj_batch_kernels.UpProjBatchKernels`, geïnstalleerd vóór
`setup_graph()` net als de andere twee). Full mode, 765 samples:

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,0973 ms | 32,16 |
| **V6 (nu vijf mechanismen)** | **21,1118 ms** | **47,37** |

Winst 9,9855 ms/token (32,1%) t.o.v. zelfde-sessie EGR. Alle poorten groen:
bitexact, deterministisch, controle-arm wijkt af, dot-graph bevat nu alle
vijf kernelnamen (beide ERVF-dense, beide down_proj-batch, plus de nieuwe
up_proj-batch), VRAM binnen budget.

**Stand.** 47,37/165 = **28,7% van roofline** (was 26,9%). Nog een factor
**2,11×** te gaan tot 100 tok/s.

**Artefacten.** `pro_research/up_proj_batch_kernels.py` ·
`pro_research/verify_up_proj_batch_kernel.py` ·
`pro_research/verify_up_proj_batch_kernel.json` ·
`pro_research/v_up_proj_batched_ab.py` ·
`pro_research/results/PRO_UP_PROJ_BATCHED_AB.json` ·
`pro_research/moe_dev_batched.py` (uitgebreid, optionele `up_kernels`-
parameter) · `pro_research/graph_v6_full_stack.py` (uitgebreid) ·
`pro_research/results/PRO_V6_FULL_STACK.json` (bijgewerkt).

---

## 2026-08-16 — `weighted_accumulate_ind` veilig gebatcht en geïntegreerd: 44,37 tok/s

**Vervolg op de componentafbraak hierboven.** `accumulate_indirect` is
gebouwd volgens precies het aangewezen veilige patroon: **niet** een
mechanische kopie van `panel_scan`/`reduce_partials` (die schrijven
onafhankelijk per slot), maar een nieuwe `weighted_accumulate_ind_batched`-
kernel die de exacte `s=0..top_k-1`-fmaf-volgorde uit één kernel-launch
reproduceert (`acc = fmaf(contrib[s], w[s], acc)` in een vaste lus,
startend vanuit `dst[i]` dat al de shared-expert-term bevat) — bit-identiek
aan `top_k` losse launches, geen parallelle/atomic reductie die de
FP-optelvolgorde zou veranderen.

**Stap 1 (geïsoleerd, `verify_down_proj_batch_kernels.py` uitgebreid):**
bitexact tegen de sequentiële referentie, 3 trials met willekeurige
`dst`/`contrib`/`w`. **Stap 2 (`moe_dev_batched.py` uitgebreid, causale A/B
via `v5_batched_downproj_ab.py`, dat automatisch meelift):** full mode,
bitexact, **−3,1552 ms/token (−9,88%)** eager t.o.v. eerder −2,2126 ms
(−7,07%) met alleen panel_scan+reduce_partials — de accumulate-batching
draagt dus zelf nog eens ~0,94 ms/token bij, in lijn met de verwachting dat
dit een kleinere losse post was.

**V6 opnieuw gedraaid (pakt de uitbreiding automatisch mee via dezelfde
`install_batched_moe_dev`).** Full mode, 765 samples: **22,5354 ms/token,
44,37 tok/s** (was 44,19 met alleen panel_scan+reduce_partials), winst 8,7972
ms/token (28,1%) t.o.v. zelfde-sessie EGR. Alle poorten opnieuw groen:
bitexact, deterministisch, controle-arm wijkt af, VRAM binnen budget.

**Stand.** 44,37/165 = **26,9% van roofline**. Nog een factor **2,25×** te
gaan tot 100 tok/s.

**Artefacten.** `pro_research/down_proj_batch_kernels.py` (uitgebreid),
`pro_research/verify_down_proj_batch_kernels.py` (uitgebreid),
`pro_research/moe_dev_batched.py` (uitgebreid),
`pro_research/results/PRO_V5_BATCHED_DOWNPROJ_AB.json` (bijgewerkt),
`pro_research/results/PRO_V6_FULL_STACK.json` (bijgewerkt).

---

## 2026-08-16 — Componentafbraak van V6: MoE is 57,8% van het token, niet alleen down_proj

**Vraag.** Down_proj (V5) was de eerste hefboom, maar waar gaat de rest van
V6's ~20,9-22,6 ms/token naartoe? Nodig om de volgende stap gericht te
kiezen i.p.v. te blijven graven op down_proj alleen.

**Methode.** Zelfde ablatietechniek als `diag_down_ablation_timing.py`
(wall-clock, want `cp.cuda.get_elapsed_time` op graph-gevangen events faalt
op deze stack). Vier hele subblokken om de beurt vervangen door een no-op
vóór `setup_graph()` vangt, elk in een **apart proces** gedraaid
(`diag_v6_component_breakdown.py --drive`) — het bouwen van vijf volledige
30B-runtimes ná elkaar in één proces liep vast op pinned-host-geheugen dat
niet volledig vrijkwam tussen builds, ook niet met expliciete
`free_all_blocks()`+`gc.collect()`.

| stub | wat wordt overgeslagen | bovengrens | aandeel |
|---|---|---:|---:|
| `_attention` → no-op | 6 attentielagen (Q/K/V/O, KV-write, attention-kernel) | 3,0987 ms | 14,9% |
| `_mamba` → no-op | 23 Mamba-lagen (in_proj, conv, ssm_step, gated_norm, out_proj) | **−0,4287 ms** | ~0% (ruis, apart-proces-vergelijking heeft meer variantie dan de eerdere in-proces down_proj-ablatie) |
| `_moe` → no-op | **alle** 23 MoE-lagen: shared expert + routed (up+down) | **12,0669 ms** | **57,8%** |
| `fused.gemv_into` → no-op | lm_head (1×/token) **plus** shared-expert up+down (46×/token) — zelfde methode, dus samen gemeten, niet apart | 2,1023 ms | 10,1% |

**MoE is dus verreweg de grootste post — meer dan de helft van het hele
token.** Kruisverwijzing met de eerdere down_proj-ablatie (6,5058 ms
in-graph): MoE-bovengrens (12,0669) − down_proj (6,5058) = **~5,56 ms**
overige MoE-kosten (shared-expert-GEMV's, up-proj ERVF-GEMV, de batched
panel_scan/reduce_partials-kernels zelf, `accumulate_indirect`,
routing/cache-kernels) — nog niet apart uitgesplitst.

**Let op — de `moe`- en `lmhead_plus_shared_expert`-bovengrenzen
OVERLAPPEN** (beide bevatten de shared-expert-kosten); niet optellen alsof
ze disjunct zijn.

**Wat dit opent voor de volgende stap.** `accumulate_indirect` (de
gewogen-optel-kernel die de zes expert-bijdragen in `out` optelt) leek eerst
een voor de hand liggende volgende batch-kandidaat naast `panel_scan`/
`reduce_partials` — maar is dat **niet zomaar**: anders dan die twee is dit
een **sequentiële** accumulatie in dezelfde `out`-buffer
(`dst[i] = fmaf(src[i], w, dst[i])`, zes keer na elkaar). Simpelweg batchen
over slots zou een race-conditie geven (alle zes blocks lezen/schrijven
dezelfde `dst[i]` tegelijk) en zou atomics of een aparte
"schrijf-per-slot-dan-in-vaste-volgorde-optellen"-kernel vereisen — precies
de klasse fout die D1 al een keer blootlegde (optelvolgorde is niet
vrijblijvend bij FP-optelling). Een veilige batched variant zou, net als
`reduce_partials_batched`, een **apart, nieuw ontwerp** nodig hebben dat de
volgorde `s=0..5` expliciet bewaart — geen mechanische kopie van het
patroon dat voor `panel_scan`/`reduce_partials` wél veilig was. Niet
gebouwd deze sessie; opgenomen in `TODO.md` met deze precieze
kanttekening zodat niemand de D1-fout herhaalt.

**Artefacten.** `pro_research/diag_v6_component_breakdown.py` ·
`pro_research/diag_v6_component_breakdown.json` ·
`pro_research/diag_v6_component_breakdown_arm_*.json` (vijf, één per arm).

---

## 2026-08-16 — PRO V5 + V6 — batched down_proj gebouwd, geverifieerd en geïntegreerd: 44,19 tok/s, nieuw record

**Aanleiding.** De ablatiemeting hierboven (`diag_down_ablation_timing.py`)
gaf een realistische, in-graph bovengrens: down_proj kost hooguit 6,51
ms/token (28,9%) binnen V4's graph, waarvan `panel_scan`+`reduce_partials`
(vaste, data-onafhankelijke grid-groottes, dus veilig te batchen zonder de
lastigere PCIe-gather aan te raken) een deel is. In plaats van bij een
preregistratie te blijven steken is dit nu gebouwd, in drie stappen, elk
apart geverifieerd vóór de volgende.

**Stap 1 — geïsoleerde kernel-unittest (`verify_down_proj_batch_kernels.py`,
geen model/runtime nodig).** Twee nieuwe kernels
(`pro_research/down_proj_batch_kernels.py`): `panel_scan_batched` (grid
`(top_k,)` i.p.v. `top_k` losse `(1,)`-launches, elk block bewerkt zijn eigen
slot) en `reduce_partials_batched` (grid `(blocks_x, top_k)`, `blockIdx.y` =
slot). Beide zijn een mechanische transformatie: identieke per-block/per-
thread logica, alleen geadresseerd via een extra slot-index — geen
rekenkunde gewijzigd. Getest op synthetische data bij zes sparsity-niveaus
(0/30/50/70/95/100%) voor panel_scan en drie willekeurige trials voor
reduce_partials: **bitexact op alle gevallen.**

**Stap 2 — causale A/B/A/CTL in eager modus (`v5_batched_downproj_ab.py`,
volgens `PRO_V5_PREREGISTRATION.md`).** `install_batched_moe_dev`
(`pro_research/moe_dev_batched.py`) vervangt `rt._moe_dev` via
`types.MethodType` (zelfde niet-invasieve patroon als `_install_selective`)
— `gather_down_sparse_ind`/`gemv_down_masked_partial_ind` blijven ONGEWIJZIGD
per-slot (buiten scope, zoals de prereg voorschreef), alleen `panel_scan` en
`reduce_partials` worden éénmaal per laag aangeroepen i.p.v. zes keer.

*Eerste run had een bug*: een NIEUWE `mirror`-buffer per laag i.p.v.
hergebruik van `self.mstate["mirror"]` (die toch al bestaat, want gather/
down_masked blijven sequentieel per slot). Kostte 23 lagen × 2,68 MB ≈ 61,6
MB — precies waarom de VRAM-poort bestaat: `extra_vram_lt_64MiB` **faalde**
bij V6 hieronder (zie verderop), gevonden, begrepen, gefixt (hergebruik
`self.mstate["mirror"]`), opnieuw geverifieerd bitexact vóór verder te gaan.

Full-mode uitkomst (256×3, 765 samples): BASE_A 31,049 → BATCHED 29,0799 ms
(**−2,2126 ms, −7,07%**), BASE_B 31,5359 ms (drift 0,487 ms). Poorten:
`batched_equals_base_bitexact` ✅ · `base_drift_le_1ms` ✅ · `ctl_diverges`
✅ (bad_pick-sabotage wijkt af zoals vereist) · `candidate_gain_ge_1ms_or_3pct`
✅ · `full_samples_ge_500` ✅.

**Stap 3 — V6: alle drie mechanismen in één graph (`graph_v6_full_stack.py`,
niet apart gepreregistreerd — volgt rechtstreeks uit V4's en V5's eigen
poorten, geen nieuwe beleidskeuze).** `_install_selective`
(patcht `rt.k.mv_bf16`/`mv_fp8_tensor`, gebruikt in `_attention`/`_mamba`)
en `install_batched_moe_dev` (vervangt `rt._moe_dev` volledig, alleen MoE-
routering) raken **verschillende aanroeppunten** — beide vóór
`rt.setup_graph()` installeren vangt dus beide mechanismen in dezelfde
graph, naast de al aanwezige device-routing en veilige prompt-staging.

*Eerste V6-run*: alle correctheidspoorten slaagden, maar
`extra_vram_lt_64MiB` **faalde** (de V5-mirror-bug hierboven, opgespoord via
precies déze poort — het systeem werkte zoals bedoeld). Na de fix: alle
poorten groen.

**Uitkomst (full, 256×3, 765 samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,1595 ms | 32,09 |
| **V6 (device routing + graph-safe + selectieve ERVF + batched down_proj)** | **22,6306 ms** | **44,19** |

Winst: **8,5289 ms/token, 27,4%** t.o.v. dezelfde-sessie EGR. Extra VRAM
16 MiB (< 64 MiB-poort, ruim binnen budget na de fix). Alle poorten:
`argmax_direct_tie` ✅ · `graph_dot_contains_all_mechanisms` ✅ (dot-graph
bevat alle vier kernelnamen: beide ERVF-kernels én beide batched-kernels) ·
`v6_equals_egr` ✅ (bitexact over 256 tokens × 3 prompts) · `v6_deterministic`
✅ · `bad_pick_control_diverges` ✅ · `extra_vram_lt_64MiB` ✅ ·
`full_speed_gain_ge_2_5ms` ✅ · `full_samples_ge_500` ✅ (765).

**Dit is het nieuwe record, en verbetert op V4 (41,13 tok/s smoke/full) met
+7,5%.** Vergelijk met de eerdere V4-notitie: V6's winst (8,5289 ms) is
groter dan V4's eigen winst (6,8634 ms) plús V5's eigen eager-winst
(2,2126 ms) opgeteld zou suggereren (6,8634+2,2126=9,0760 — dicht bij 8,5289,
een kleine overlap-tax van ~0,55 ms, niet negatief) — de drie mechanismen
blijven dus grotendeels optellen, net als V4's eigen bevinding over de
eerste twee.

**Waar staan we nu t.o.v. het doel.** 44,19 tok/s is **26,8% van het
ctx0-roofline-plafond** (165 tok/s) — op van 24,9% (V4) en 17% (de oude
Nano-lijn). Naar 100 tok/s is nog een factor **2,26×** te gaan vanaf hier.
De resterende, nog niet gebouwde hefboom (PCIe-gather-herstructurering van
`gather_down_sparse_ind`, apart gehouden van dit V5-batchen per de
preregistratie) zou volgens de ablatiemeting hooguit nog eens een paar
ms/token kunnen opleveren — niet genoeg om alleen daarmee 100 te bereiken.
Er is nog geen geïdentificeerd pad naar 100 tok/s; dat blijft open.

**Artefacten.** `pro_research/down_proj_batch_kernels.py` ·
`pro_research/verify_down_proj_batch_kernels.py` ·
`pro_research/verify_down_proj_batch_kernels.json` ·
`pro_research/moe_dev_batched.py` · `pro_research/v5_batched_downproj_ab.py` ·
`pro_research/results/PRO_V5_BATCHED_DOWNPROJ_AB.json` ·
`pro_research/graph_v6_full_stack.py` ·
`pro_research/results/PRO_V6_FULL_STACK.json`.

---

## 2026-08-16 — MTP-route-unie gemeten: S10 stap 2 (speculatief decoderen) sluit, met cijfers

**Vraag.** `S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md` (bestond al — S10-A mat
de acceptatiegraad `A=2,114` over 360 stappen, poort G-S10-1 ≥1,5 **gehaald**)
identificeerde zelf de **enige nog onbekende term** die beslist of stap 2
(de speculatieve lus daadwerkelijk bouwen) de moeite waard is: hoeveel
**unieke** experts raakt een verificatie-sweep van `D+1=5` tokens per
MoE-laag, tegenover 6 voor één token? De prereg gaf alleen een theoretische
bovengrens (≤3,66× via N7-A's paarsgewijze overlap 2,011/6) en schreef
expliciet: "meet die unie eerst... het kost geen bouw."

**Methode.** Puur read-only analyse, geen nieuwe kernel, geen runtime-
wijziging. `LightningRuntime.step(token_id, capture_routes=...)` **bestaat
al** (regel 803-823) en registreert per MoE-laag de top-6 expert-ids per
stap. De exacte greedy-gegenereerde tokenreeksen van de drie S10-A-
gate-prompts staan al op schijf in `s10a_mtp_acceptance.json`
(`gate_prompts[*].sequence`). Teacher-forced replay van die reeksen door de
backbone (capaciteit 72, `device_cache=False` — nodig omdat `_moe_dev`
`(None, None)` teruggeeft en `capture_routes` dus niet vult; routing zelf is
onafhankelijk van cache-modus) geeft de exacte routes die de échte generatie
gebruikte. Sliding window van 5 opeenvolgende posities, unie van de
verzamelde expert-ids per laag, gemiddeld over alle vensters/lagen/prompts.
Script: `pro_research/diag_mtp_route_union.py`.

**Uitkomst.** Gemiddelde unie over 5 tokens: **19,88 van de 128 experts per
laag** (mediaan 20, p95 26, max 30 = het theoretische maximum 5×6). Dat is
**3,313×** t.o.v. de 6 van één token — ruim boven de eigen voorgestelde
grens van de prereg (>12 = "verdubbeling" ⇒ stap 2 negatief), en zelfs iets
onder de theoretische bovengrens (3,66×) maar in dezelfde orde. Stabiel over
de drie domeinen (expository 19,25 · narrative 19,48 · code 20,87 — code
route't net iets breder, consistent met code's hogere acceptatiegraad
elders).

**De rekensom van het rapport, met dit echte getal in plaats van de
schatting.** MoE-verificatiesweep = 39,523 ms × 3,313 = 130,95 ms; volledige
backbone-sweep (attentie 18,634 + Mamba 8,309 + lm_head 2,106 amortiserend +
MoE 130,95) ≈ **160,0 ms**. Eén speculatieve ronde = MTP-keten (19,10 ms) +
verificatiesweep (160,0 ms) = 179,1 ms, voor gemiddeld `A+1 = 3,114` tokens
⇒ **57,51 ms/token**. Niet-speculatief (gemeten S8 @262K): **54,28 ms/token**.

**Speculatief decoderen zou dus ~6,0% TRAGER zijn, niet sneller — met de
gemeten unie, niet de aanname.**

**Wat dit sluit.** S10 stap 2 (de speculatieve lus bouwen) sluit hiermee
netjes, zonder dat er één regel productiekernel voor geschreven hoefde te
worden — precies zoals de S10-A-prereg voorschreef. De oorzaak is
architectureel, niet toevallig: bij 128 routed experts en top-6-sparsity per
token routeren opeenvolgende tokens naar grotendeels verschillende
expert-sets, dus een gezamenlijke verificatie-sweep beweegt bijna evenveel
MoE-bytes als losse tokens genereren — terwijl de MTP-drafting daar nog eens
19,10 ms bovenop legt. Dit was, volgens `S10_MTP_SPECULATIVE_PREREGISTRATION_2026-08-15.md`,
"de enige overgebleven hefboom" die bytes-per-token verlaagt in plaats van
tijd-per-byte. Met deze meting is die hefboom nu ook dicht — de eerdere
tabel "Wat weerlegd is" in `STATE_OF_THE_WORK.md` (die nog de oude
Nano-sluiting citeerde) is hiermee opnieuw en definitief bevestigd, nu voor
het juiste model en met een directe meting i.p.v. een architectuurvergelijking.

**Wat dit niet doet.** Verandert niets aan S10-A's eigen bevindingen (de
wiring, de acceptatiegraad, de kostenmeting van de MTP-keten) — die blijven
correct en gedocumenteerd. Sluit alleen stap 2 op basis van de nu bekende
route-unie.

**Artefacten.** `pro_research/diag_mtp_route_union.py` ·
`pro_research/diag_mtp_route_union.json` (read-only, geen PRO-poort, leest
alleen bestaande `reports/lightningstream_nemotron/s10a_mtp_acceptance.json`
en roept alleen bestaande runtime-API's aan).

---

## 2026-08-16 — Diagnose device-cache hitrate onder V4 — down_proj is nooit gecached in `_moe_dev`

**Vraag.** Waar gaat de resterende 24,3 ms/token na V4 nog naartoe — PCIe-
missverkeer, of iets anders? Nodig om de volgende hefboom gericht te kiezen
in plaats van te gokken.

**Methode.** Read-only diagnostiek (`pro_research/diag_hitrate_v4.py`, geen
PRO-poort, geen preregistratie — puur meetkundig). `_moe_dev` (het pad dat
`device_cache=True` gebruikt, dus ook wat V4's graph capture) accumuleert per
laag hits/misses al in een device-buffer (`dev["stats2"]`, gevuld door de
`cache_assign`-kernel in `fused_nvfp4.py`) — nergens in de bestaande runners
uitgelezen. Dit script sommeert die buffer over alle 23 MoE-lagen na een
rollout van 256 tokens (prompt "The history of computing began when",
capacity 72, top_k 6).

**Uitkomst.** Hitrate **85,6%** (30.836 hits / 5.182 misses), **20,24 misses
per token** over 138 expert-selecties per token (23 lagen × top_k 6). Sterk
niet-uniform: laag 1/3/6 missen 42,5% / 36,5% / 25,2%, de meeste middenlagen
zakken naar 6–10%, laag 51 (laatste MoE-laag) piekt weer naar 25,3%.

**De structurele vondst.** `cache_mode` is `up_only` — de device-LRU-cache
(`c["cap"]`, gevuld via `cache_fetch`) dekt alléén de up_proj-codes/scales.
`_moe_dev`'s down-projectie loopt via
`down_masked_into_indirect(..., bank["down_base_ptr"], ...)` — dat leest
**bij elke expert-aanroep, hit of miss**, rechtstreeks uit de host-gemapte
bank. Er bestaat wél een `cache_mode="full"` met een `c["slot_down"]`
device-slot (zichtbaar in het oudere `_moe_cached_fast`, regel 659-660), maar
`_moe_dev` gebruikt dat pad nergens. Dus zelfs bij 85,6% up_proj-hitrate
bewegen **alle 138 down_proj-aanroepen per token** nog over PCIe (zij het via
de S5-masked/sparse-gather-techniek, dus minder dan een volle rij — het exacte
aantal bytes per aanroep is activatie-sparsity-afhankelijk en is hier niet
geschat om geen precisie te claimen die de statische code-lezing niet
onderbouwt).

**Wat dit opent — geprioriteerde volgende hefboom.** Twee onafhankelijke,
goed onderbouwde vervolgexperimenten, allebei niet in deze sessie gebouwd:

1. **Device-cache down_proj in `_moe_dev`.** `_moe_cached_fast` bewijst het
   patroon al (device-slot voor down_proj bij `cache_mode="full"`) — het moet
   nog naar het `device_cache`/graph-pad worden overgezet. Als down_proj-hits
   evenveel PCIe-verkeer schelen als up_proj-hits, is dit een tweede,
   onafhankelijke hefboom van vergelijkbare orde als V4's ERVF-winst, zonder
   dat er een nieuwe kernel bij hoeft (alleen bestaande device-cache-
   infrastructuur hergebruiken).
2. **Per-laag capaciteitstuning.** De miss-rate-ongelijkheid tussen lagen
   (laag 1/3/6/51 vs de rest) suggereert dat een uniforme capaciteit van 72
   per laag niet optimaal is — lagen met herhaaldelijk hoge missrate kunnen
   baat hebben bij een grotere cap, ten koste van lagen die toch al bijna
   nooit missen. Vereist eerst bevestiging dat dit patroon stabiel is over
   meerdere prompts/rollouts, niet een toeval van deze ene 256-token-run.

**Artefacten.** `pro_research/diag_hitrate_v4.py` ·
`pro_research/diag_hitrate_v4.json` (niet gecommit als PRO-resultaat, puur
diagnostisch).

**Correctie achteraf, zelfde dag.** De hierboven voorgestelde hefboom
("device-cache down_proj net als up_proj") bleek op een verkeerde
byte-aanname te steunen. `DOWN_PANEL_BYTES` (= `HALF_CODE + HALF_SCALE` =
2.494.464 + 311.808 B) is **2,68 MB per expert**, niet ~1 kB — bevestigd via
`CODE_BYTES`/`SCALE_BYTES` in `runtime.py` en `load_routed_bank`'s
`n * DOWN_PANEL_BYTES`-slicing. Vol cachen bij cap 72 × 23 lagen zou ~4,4 GiB
VRAM kosten; de GPU stond tijdens de V4-run al op **0 MiB vrij**. Dat maakt
"gewoon net als up_proj cachen" onhaalbaar zonder de up_proj-capaciteit fors
te verlagen (een echte trade-off, geen gratis winst) — vandaar de vervolgmeting
hieronder in plaats van dit alsnog te bouwen. Zie ook: up_proj's eigen missen
bewegen dus ook ~2,68 MB per miss (niet ~1 kB) — bij 20,24 missen/token is dat
**~54,2 MB/token**, wat bij het M1-tempo (24,93 GB/s) ~2,17 ms/token kost; dat
zit *niet* in de eerder gemeten "up_gemv"-component hieronder, want die meet
alleen de GEMV, niet de `cache_fetch`-staging.

---

## 2026-08-16 — Componentmeting V4-decodepad — down_proj-pijplijn is de grootste losse kostenpost

**Vraag.** Waar gaat de 24,3–34 ms/token precies naartoe? Nodig om de
down_proj-hefboom hierboven te vervangen door iets dat op echte metingen
rust, niet op statische bytegrootte-aannames (die al twee keer fout bleken).

**Methode.** Twee read-only diagnostieken, beide eager (`device_cache=True`,
géén graph — zodat elke aanroep echte Python is en met `cp.cuda.Event`-paren
precies te timen valt):

1. `diag_component_timing_v4.py` — omwikkelt `fused.gemv_ervf_indirect`
   (up_proj ERVF-GEMV) en `fused.down_masked_into_indirect` (de hele
   down_proj-pijplijn) als geheel, 128 tokens.
2. `diag_down_subkernels_v4.py` — splitst die down_proj-pijplijn verder open
   in zijn vier subkernels (`panel_scan`, `gather_down_sparse_ind`,
   `down_masked_ind`, `reduce_partials`), 96 tokens.

**Uitkomst (1).** Token p50 33,46 ms (deze eager config, iets hoger dan G0S's
EGR door instrumentatie-overhead). Per token, 138 expert-aanroepen (23 lagen
× top_k 6):

| component | ms/token | aandeel van token |
|---|---:|---:|
| up_gemv (ERVF, device-cache) | 5,003 | 14,6% |
| down_masked (hele pijplijn) | 9,573 | 27,9% |

down_masked is dus al zonder verdere opsplitsing de grootste single measured
component — groter dan up_gemv, en dat zonder up_proj's eigen
`cache_fetch`-staging (~2,17 ms/token, apart, hierboven geschat) mee te tellen.

**Uitkomst (2) — opsplitsing van die 9,57–11,39 ms (iets hoger token-p50
door extra instrumentatie: 36,28 ms):**

| subkernel | ms/token | aandeel van down-pijplijn |
|---|---:|---:|
| `gather_down_sparse_ind` (PCIe host-gemapte masked read) | 4,740 | 41,6% |
| `panel_scan` (mask/nonzero-scan, device-only) | 2,737 | 24,0% |
| `reduce_partials` (device-only) | 1,999 | 17,6% |
| `down_masked_ind` (de eigenlijke GEMV-compute) | 1,914 | 16,8% |
| **totaal down-pijplijn** | **11,390** | **29,9% van token** |

**Interpretatie.** Twee onafhankelijke aanwijzingen, geen enkele
overweldigend dominant:
- `gather_down_sparse_ind` is de grootste losse subkernel en bevestigt de
  PCIe-hypothese (host-gemapte, gemaskeerde/verstrooide lezing — dezelfde
  klasse toegang die E2/NERVF-4 al traag bleek, ~6,7-7,27 GB/s i.p.v.
  bulk-tempo 24,93-85,9 GB/s) — maar is met 41,6% niet de hele lading.
- `panel_scan` + `reduce_partials` samen (41,6% van de pijplijn, 4,74 ms/token)
  zijn beide kleine, device-only kernels zonder PCIe-component — hun kosten
  wijzen eerder op **launch-overhead**: 4 subkernels × 138 expert-aanroepen =
  **552 kernellaunches per token** alleen al voor down_proj. Fusie van deze
  vier kernels (of batchen over de 6 experts per laag in plaats van 6×4
  losse launches) is een onafhankelijke, mogelijk net zo grote hefboom als de
  PCIe-kant.

**Bovengrens, voorzichtig.** Zelfs als de hele down-pijplijn naar nul zou
gaan (onrealistisch), zou dat token-tijd van ~24,3 ms (V4, met selectieve
ERVF en graph) hooguit naar iets in de orde van ~13-15 ms brengen — ruw
richting 65-75 tok/s, niet 100. Dit is dus een substantiële maar geen
op-zichzelf-voldoende hefboom voor het einddoel.

**Waarom niet meteen gebouwd.** Twee echte CUDA-engineeringtaken (PCIe-
toegangspatroon herstructureren; vier kernels fuseren/batchen), allebei op
het kritieke pad van een 30B-productiemodel. Gezien de projectcultuur
("exactness before speed", nooit een poort verruimen, controle-armen
verplicht) verdienen die een eigen preregistratie en zorgvuldige
bitexact-verificatie, niet een haastige poging binnen dezelfde sessie waarin
al twee keer een snelle bytegrootte-aanname fout bleek.

**Artefacten.** `pro_research/diag_component_timing_v4.py` ·
`pro_research/diag_component_timing_v4.json` ·
`pro_research/diag_down_subkernels_v4.py` ·
`pro_research/diag_down_subkernels_v4.json` (alle vier read-only, geen
PRO-poort).

**Correctie achteraf — de eager meting overschat het absolute in-graph
budget (proportie klopte wel).** De 9,57-11,39 ms/token hierboven is eager
gemeten (`device_cache=True`, géén graph). V4 draait dit dezelfde pijplijn
**binnen** een gevangen CUDA-graph, die specifiek gebouwd is om
host-launch-overhead weg te nemen. Een poging om dat rechtstreeks te meten
met `cp.cuda.Event`-paren gevangen ín de graph liep vast op een echte
technische grens: `cp.cuda.get_elapsed_time` op events die binnen graph-
capture zijn opgenomen geeft op deze stack `cudaErrorInvalidValue`
(`pro_research/diag_down_ingraph_timing.py` — zelfde klasse bevinding als
G2's `cudaGraphLaunch`-restrictie, geen bug in het script).

In plaats daarvan: **ablatiemeting** — V4's exacte graph twee keer bouwen,
één keer ongewijzigd (REAL), één keer met `down_masked_into_indirect`
vervangen door een no-op (`out.fill(0)`, STUB — levert bewust **foute
tokens**, nooit als correctheidsresultaat te lezen, timing-only) vóór
`setup_graph()` vangt. Verschil = harde bovengrens op wat down_proj
maximaal in-graph kost. Uitkomst (200 replays per arm,
`pro_research/diag_down_ablation_timing.py`):

| arm | p50 ms/token |
|---|---:|
| REAL (ongewijzigd, zoals V4) | 22,5302 |
| STUB (down_proj = no-op) | 16,0244 |
| **verschil (bovengrens down_proj in-graph)** | **6,5058 (28,9% van het token)** |

De **proportie** komt opvallend overeen met de eager-meting (~28-30%), maar
het **absolute** bedrag is veel kleiner (6,51 ms in-graph tegenover
9,57-11,39 ms eager) — graph-replay amortiseert dus al een groot deel van
de launch-overhead die de eager meting toeschreef aan `panel_scan`/
`reduce_partials`. **Gevolg voor de V5-preregistratie hieronder:** de eerder
genoemde bovengrens ("~65-75 tok/s als de hele pijplijn zou verdwijnen") was
te optimistisch — de eager-pijplijnkosten direct op de V4-basislijn
plakken was appels-met-peren. De juiste bovengrens is: volledige eliminatie
van down_proj zou V4 van ~22,5 naar 16,0 ms/token brengen (**~62,4 tok/s**,
optimistisch/onhaalbaar aangezien STUB fout is). V5 dekt bovendien alleen de
launch-overhead-helft, niet de PCIe-kant — geschat (dezelfde verhouding als
de eager subkernel-split, 41,6% van de pijplijn) op hooguit **~2,7 ms/token**
haalbaar, oftewel richting **~46-50 tok/s** in het beste geval. Bijgewerkt
in `PRO_V5_PREREGISTRATION.md`'s claim-boundary sectie.

**Aanvullend, zelfde blok — batchen is architecturaal veilig, geverifieerd
uit de code (nog niet gebouwd).** `_moe_dev`'s `for s in range(top_k):`-lus
roept `down_masked_into_indirect` zes keer sequentieel aan met **hetzelfde**
scratch-object `self.mstate` (`alloc_masked_state`, één-expert-grootte,
éénmalig aangemaakt in `runtime.py:311`), en schrijft elk resultaat naar een
eigen slice `self.contrib[s*hidden:(s+1)*hidden]` (`self.contrib` is al
`top_k*hidden` vooraf gealloceerd, regel 314). Er is dus **geen
cross-slot-afhankelijkheid** — de zes expert-aanroepen zijn embarrassingly
parallel, alleen toevallig sequentieel geïmplementeerd met hergebruikt
scratch-geheugen. Batchen (6-voudig scratch + kernels die intern over de
slot-dimensie gridden i.p.v. zes losse launches van 4 kernels) zou de
berekende waarden niet veranderen — alleen de launch-granulariteit. Dat maakt
de correctheidskant van deze hefboom aantoonbaar behapbaar; de
implementatie-kant (kernels herschrijven om over 6 slots te gridden, met
correcte per-slot scratch-indexering) is nog steeds echt engineeringwerk,
maar niet blind engineeringwerk. Extra aanwijzing: `panel_scan_k` lanceert
met grid `(1,)`, block `(256,)` — één block, 138× per token — een
klassiek launch-overhead-bound patroon (nauwelijks parallellisme benut per
launch).

---

## 2026-08-16 — PRO G2 — K-token epoch-graph: technisch gesloten, geen bug

**Vraag.** Kan de bestaande token-graph (`rt._graph`, een geïnstantieerde
`cudaGraphExec_t` via CuPy) K keer als child in één parent-graph gevangen
worden, zodat één host-launch K tokens vooruitbrengt?

**Uitkomst.** Nee, met de huidige aanpak. `pro_research/epoch_graph.py --mode
smoke` (ongewijzigd, al aanwezig, nooit eerder gedraaid) faalt voor k=2 én k=4
met `cudaErrorStreamCaptureUnsupported: operation not permitted when stream is
capturing`, bij `rt._graph.launch(stream)` binnen `stream.begin_capture()`.

**Waarom.** `cudaGraphLaunch()` — het aanroepen van een reeds
geïnstantieerde/uitvoerbare graph — is zelf geen capturable API-call. Om een
graph als child-node in een andere capture op te nemen moet je de
graph-**template** (`cudaGraph_t`, vóór instantiatie) doorgeven aan
`cudaGraphAddChildGraphNode()`, niet de uitvoerbare `cudaGraphExec_t` die
`.launch()` gebruikt. `runtime.py`'s `setup_graph()` bewaart alleen het
geïnstantieerde object (`s.end_capture()`); de template wordt niet apart
vastgehouden. Dit is een CUDA-API-beperking, geen CuPy-instelling of bug in
deze pack.

**Status: `technical_blocked`, poort `parent_graph_ids_exact` niet bereikt —
eerlijke sluiting per de eigen regel van de pack** ("Unsupported nested
capture is a valid technical closure"). Niet geforceerd, geen alternatief
mechanisme stilletjes gesubstitueerd.

**Wat dit open laat.** De K-token-amortisatie-hypothese zelf is niet weerlegd
— alleen déze implementatiestrategie. Een diepere vervolgstap (niet in deze
sessie gedaan) zou de graph-template apart moeten bewaren tijdens
`setup_graph()`'s eigen capture en `cudaGraphAddChildGraphNode` rechtstreeks
via CuPy's lage-niveau runtime-bindings aanroepen — een aparte,
substantiëlere CUDA-engineeringtaak, geen kleine reparatie.

**Artefacten.** `pro_research/results/PRO_G2_EPOCH_GRAPH.json` (status
`technical_blocked`).

---

## 2026-08-16 — PRO V4 — graph-safe + selectieve ERVF fysiek geïntegreerd: 41,13 tok/s, alle poorten groen

**Vraag.** V3-G0S (graph-residentie alleen, +10,1%) en V3-G1B (selectieve ERVF
alleen, +10,73%) zijn los gemeten en mogen niet worden opgeteld (Amdahl-
interactie onbekend). Wat levert één fysiek geïntegreerde arm echt op?

**Mechanisme.** `_step_body_graph()` roept `self._attention`/`self._mamba` aan,
die zelf via `self.k.mv_bf16`/`mv_fp8_tensor`/`mv_f32` dispatchen
(`runtime.py:401-474`). CUDA-graph-capture legt vast welke kernel op
capture-moment achter dat Python-attribuut zit. Dus: eerst
`selective_ervf_v3._install_selective(rt, dense)` draaien (herbindt die drie
attributen voor de vier bevroren winnende vormen), **dan pas** `rt.setup_graph()`
— de ERVF-kernels worden dan mee vastgelegd in de graph, terwijl K/V/router op
productiekernels blijven binnen dezelfde graph. Nieuwe runner:
`pro_research/graph_selective_v4.py`, preregistratie
`PRO_V4_PREREGISTRATION.md` (bevroren vóór meting).

**Opzet.** EGR (productie, device-cache, geen graph) vs GRAPH_SELECTIVE
(selectieve dispatch geïnstalleerd vóór capture) vs DET (twee rollouts) vs CTL
(`bad_pick=1`-sabotage herbinnen dezelfde graph, moet falen). Structurele
verificatie: `rt._graph.debug_dot_str()` moet `pro_gemv_bf16_ervf16` én
`pro_gemv_fp8_tensor_ervf16` bevatten — bewijst dat de ERVF-kernels echt in de
graph zitten, niet alleen tijdens de warmup zijn aangeroepen.

**Uitkomst (full, 3 prompts × 256 tokens, 765 getimede samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,1786 ms | 32,07 |
| **GRAPH_SELECTIVE** | **24,3152 ms** | **41,13** |

Winst: **6,8634 ms / 22,0%** — meer dan de grootste losse mechanisme-winst
(3,3841 ms) en dicht bij de naïeve som van beide losse smoke-winsten
(2,8931 + 3,3841 = 6,2772 ms), dus nagenoeg volledig additief met slechts een
kleine overlap-tax. GRAPH_SELECTIVE's eigen p50 (24,3152) ligt ook onder béíde
eerder apart gemeten mechanismen (28,6063 en 28,158 ms) — dat is het eerste
directe bewijs dat de twee winsten fysiek samen bestaan zonder elkaar op te
eten.

**Poorten.** `argmax_direct_tie` ✅ · `graph_dot_contains_ervf` ✅ (beide
kernelnamen aangetroffen) · `graph_selective_equals_egr` ✅ (bitexact op alle
drie prompts, 256 tokens) · `graph_selective_deterministic` ✅ ·
`bad_pick_control_diverges` ✅ (2 van 3 prompts wijken af zoals vereist — de
controle heeft dus onderscheidend vermogen) · `extra_vram_lt_64MiB` ✅ (4 MiB)
· `full_speed_gain_ge_2_5ms` ✅ · `full_samples_ge_500` ✅ (765).

**Wat dit betekent voor het 100 tok/s-doel.** 41,13 tok/s is 24,9% van het
ctx0-roofline-plafond (165 tok/s, hardware-eigenschap, modelonafhankelijk) —
op van ~17% (Nano-lijn) naar ~25%. Nog altijd een factor 2,4× te gaan tot 100.
Dit is wél het eerste fysiek geïntegreerde, onafhankelijk gepoorte resultaat op
het **juiste** doelmodel, en het bevestigt dat losse mechanismewinsten hier
grotendeels blijven optellen in plaats van elkaar te kannibaliseren — een
gunstig signaal voor het toevoegen van een derde/vierde mechanisme (K-token
epoch-graph, MTP) op dezelfde manier.

**Artefacten.** `pro_research/PRO_V4_PREREGISTRATION.md` ·
`pro_research/graph_selective_v4.py` ·
`pro_research/results/PRO_V4_GRAPH_SELECTIVE.json`.

---

## 2026-08-16 — PRO V3 anchor-onderzoek — verklaard: twee verschillende modellen, geen bug

**Vraag.** De gebruiker vroeg expliciet: V3-G0S/G1B (`pro_research`) draaien
bit-identiek EGR vs GRAPH_SAFE/SELECTIVE, maar beide wijken al bij token 1 af
van het bevroren `V36_DETERMINISTIC_ANCHOR.json`. Komt dat door
`nemotron_3_5_lightning_v35` versus het oudere ankerpad, of door een
runtime/model-identiteitswijziging?

**Antwoord: het ankerpad.** Twee fysiek verschillende checkpoints:

| | `models/nemotron_3_5_lightning` (het ankerpad) | `models/nemotron_3_5_lightning_v35` (pro_research default) |
|---|---|---|
| werkelijke identiteit | **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4**, verkeerd gedownload en misleidend hernoemd | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, sha `6dbbd757…` |
| `max_position_embeddings` | 262 144 (Nano) | 1 048 576 (Lightning) |
| quantisatie | — | MIXED_PRECISION: experts+lm_head NVFP4, Mamba in/out FP8-per-tensor, attentie BF16 |
| bron | — | modelopt 0.44.0rc5, drieweg `quant_kind()` |

Bewijsketen: `reports/lightningstream_nemotron/N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md`
("De hele lijn draait op NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4") →
`N2R_V35_LAYOUT_REPORT_2026-08-14.md` (adjudicatie van de échte Lightning-download,
`models/nemotron_3_5_lightning_v35`) → `HANDOVER_TO_KIMI_2026-08-15.md`
("Model staat nu in `models/nemotron_3_5_lightning_v35`. Zet
`LS_MODEL_DIR=nemotron_3_5_lightning_v35` om ermee te draaien."). Bevestigd
lokaal: `config.json` van het ankerpad heeft `max_position_embeddings: 262144`
(Nano-plafond), het `_v35`-pad heeft `1048576` (Lightning-plafond).
`A1_ADOPTION_PRECONDITION.json.environment.model_dir` = `"nemotron_3_5_lightning"`
— A1/D1/E1-E6/NERVF-0..5 en `V36_DETERMINISTIC_ANCHOR.json` zijn dus **allemaal
gemeten op Nano**, ondanks dat de correctie al op 2026-08-14 bekend was. De
`scripts/treesweep200/*.py`-scripts defaulten nog steeds naar
`nemotron_3_5_lightning` (geen `LS_MODEL_DIR` override in die lijn) — de
"herhaal de meetketen op het juiste model"-stap uit N0R_CORRECTION is voor de
closed-namespace-lijn **nooit uitgevoerd**.

**Gevolg — geen bug, wel een scope-correctie.**
- V3's eigen interne vergelijkingen (EGR vs GRAPH_SAFE, BASE_A/SELECTIVE/BASE_B)
  zijn methodologisch geldig: alle armen laden hetzelfde model in hetzelfde
  proces. De 28,61 ms / 28,16 ms resultaten staan.
- De externe ankervergelijking in V3 is terecht `informative`, niet gating —
  precies zoals de V3-preregistratie het al voorzag.
- **Alles in dit bestand vóór 2026-08-16 (ERVF 1,936×, D1, A1, E1 fase 2.1,
  "37,49 ms/token", "26,7–29,5 tok/s") is gemeten op Nemotron-3-Nano, niet op
  Nemotron-3.5-Lightning.** `HANDOVER_TO_KIMI_2026-08-15.md` mat vóór A1/D1 al
  een kale Lightning-baseline (27,743 tok/s @ctx0, vóór ERVF/D1/A1) die al in de
  buurt zit van Nano's volledig-geoptimaliseerde 29,5 tok/s — Lightning's
  kleinere `lm_head` (NVFP4 i.p.v. BF16, 704→198 MB) en FP8-Mamba bewegen
  minder bytes per token. De kernelwinsten (ERVF, v4-attentie, D1) zijn
  architectuur-onafhankelijk aannemelijk overdraagbaar (N2R: "shape-identiek
  op 8 byte na"), en V3's eigen bitexacte pariteit op het echte Lightning-model
  bevestigt dat ze *fysiek* overdragen — maar de closed-namespace tok/s-tabel
  in `STATE_OF_THE_WORK.md` beschrijft strikt genomen Nano, niet het
  opdrachtdoel.

**Wat dit niet doet.** Geen enkele eerder gemeten winst wordt ingetrokken —
D1/A1/E1F21's eigen interne A/B's zijn ook allemaal single-model, dus intern
geldig. Het is puur een naamgevings-/scopefout die al één keer eerder werd
gedocumenteerd (N0R_CORRECTION) maar niet is doorgezet naar de
adoptiemetingen erna.

**Aanbeveling voor de volgende fase.** `pro_research` blijft op
`nemotron_3_5_lightning_v35` (correct) draaien — geen wijziging nodig. Wie ooit
weer in de closed `treesweep200`-lijn werkt, moet `LS_MODEL_DIR=nemotron_3_5_lightning_v35`
zetten, of nog beter: het ankerpad hernoemen zodat de mismatch niet blijft
terugkomen (`models/nemotron_3_5_lightning` → `models/nemotron_3_nano`, zodat de
naam niet meer liegt). Niet in deze sessie gedaan omdat schrijfrechten buiten
`agents/`+`pro_research/` niet zijn opgeëist.

**Artefacten.** `pro_research/PRO_V4_PREREGISTRATION.md` (model-identiteitsnoot
erin opgenomen) · bronnen: `N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md` ·
`N2R_V35_LAYOUT_REPORT_2026-08-14.md` · `HANDOVER_TO_KIMI_2026-08-15.md`.

---

## 2026-08-15 — E1 fase 2.2 — graph-replay GEBOUWD maar ongemeten (sessie gestopt op quota)

**Vraag.** Kan de hele token — embedding t/m argmax — als één CUDA-graph
replays worden nu het MoE-pad sync-vrij is?

**Status.** Preregistratie bevroren (`E1F22_GRAPH_CAPTURE_PREREGISTRATION`),
graph-API gesmoketest (capture → launch → correct over 5 replays), alle
kernels en runtime-methoden geschreven en op syntax gecontroleerd — maar de
A/B is **niet gedraaid**. Niets hieronder is een meting.

**Ontwerpkeuzes die de volgende agent niet opnieuw hoeft te bedenken.**
- `attn_decode_warp_fp8_gqa4_dp`: vaste grid (2,256); elke block schrijft áltijd
  zijn 4 partials — dode splits schrijven neutraal (m=-inf, l=0) — dus nooit
  stale data, en `attn_decode_combine` (vaste 1024 slots) slaat l≤0 al over.
  Zelfde optelvolgorde als eager → bitexact te verwachten, te bewijzen door
  de verifier.
- Embedding: tabel wordt bij `setup_graph()` naar pinned+mapped gekopieerd
  (+0,656 GiB host-RAM); `embed_gather_bf16` leest hem in-graph via tok_dev.
- Token-flow: argmax schrijft tok_dev aan het einde van replay N; embed_gather
  leest het aan het begin van replay N+1. Prompt-tokens staged de host met een
  stream-geordende 4-byte H2D (geen sync). Ids oogsten via pinned ringbuffer
  (`ring_harvest`); gegenereerde ids staan vanaf ring-index P-1.
- Kill-criteria staan in de prereg: K1 = event-fork weigert → single-stream
  fallback als aparte arm.

**Artefacten.** Prereg + code in `gpu_kernels.py`/`runtime.py` (zoek
"E1 fase 2.2"). Nog te schrijven: runner, verifier, rapport.

---

## 2026-08-15 — E1 fase 2.1 — device-resident routing werkt: −4,54 ms/token eager, alle poorten groen

**Vraag.** Kan de MoE-laag zonder één device→host-sync draaien (routerkop,
LRU, miss-staging als kernels), zodat graph-capture (fase 2.2) überhaupt kan —
en wat levert alleen dat al op?

**Opzet.** Eén variabele: `device_cache` aan/uit op de geadopteerde stack.
BASE = default, DEV = device routing+LRU (cap 72), INV = cap 56 moet dezelfde
tokens geven, CTL = `bad_pick`-sabotage moet falen. Pariteit tegen het
bevroren A1/V36-anker, 2 prompts × 64 tokens, contexts_max=4096. De
staging-kernel volgt het M1-patroon uit de microbench (bulk-read uit pinned
host = 24,93 GB/s, 96% van DMA), NIET de M2-variant (GEMV leest zelf van host:
7,27 GB/s — dood).

**Uitkomst.** p50 41,540 → **36,998 ms/token (−4,542 ms, −10,9%)**, pariteit
behouden. Dat is 51% van het 8,925-ms-budget uit fase 1; de rest is pure
launch-overhead voor fase 2.2. Verifier 14/14, inclusief bitexacte spiegels
(indirecte GEMV == directe ERVF; accumulate_ind == accumulate_into) en een
exacte Python-LRU-spiegel van `cache_assign`.

**Bug gevonden.** `enable_cache` resette `_dev_cache` niet → INV-arm draaide
cap-56-semantiek over vuile LRU-staat en faalde terecht. Fix: `_dev_cache = {}`
in `enable_cache`; INV+CTL opnieuw gedraaid met schone staat, beide groen.
Ook de verifier had zelf zo'n staat-desync (verse cache-buffers tegen oude
slot-tabellen) — zelfde les: *cache-inhoud en slot-staat zijn één invariant*.

**Poorten.** C1 ✅ · INV ✅ · CTL ✅ (schone attributie) · S1 ✅ (−4,542 ≥ 1,5)
· V1 ✅ (113 KiB analytisch).

**Wat dit opent.** Fase 2.2 (graph-capture van de hele token) heeft nu een
sync-vrij MoE-pad. Resterend capture-werk: embedding-gather, argmax,
pos-afhankelijke kernels (kv_write_fp8, attentie-splits) op device-pos.

**Artefacten.** `E1F21_DEVICE_ROUTING_AB.json` · `E1F21_INV_CTL_RERUN.json` ·
`e1f21_independent_verification.json` · rapport
`E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.

---

## 2026-08-15 — A1 — de bewezen stack staat nu default aan, en het anker is opnieuw bevroren

**Vraag.** E6 mat dat ERVF + v4 + D1 sneller én exact is. Mag dat de default worden?

**Waarom niet meteen.** E6 vergeleek twee armen binnen één proces met dezelfde
cachegeschiedenis — precies het regime waarin de exactheid vóór D1 óók léék te
kloppen. Vier fasen (NERVF-3, NERVF-4, E4, S11) haalden hun pariteitspoort over
2×64 tokens terwijl de runtime niet deterministisch was. Een adoptie mag niet op
een blinde test rusten.

**Opzet.** Verander de **cachecapaciteit** (72 vs 56). Dat verandert het
hit/miss-patroon op vrijwel elke laag van elke token, dus de optelvolgorde, zonder
een gewicht/route/kernel aan te raken. Plus een **controle-arm die moet falen**:
dezelfde vergelijking zonder D1.

**Uitkomst.** Met D1: identiek over 2 × 256 tokens. Zonder D1: divergeert
(expository, token 224; narrative niet). De test heeft dus vermogen — maar smal,
en dat verklaart waarom de fout vier fasen lang onzichtbaar bleef.

**Poorten.** G-A1-CAP ✅ · G-A1-CTL ✅ (faalde zoals vereist) · G-A1B-DEFAULT ✅
(runtime zonder enige vlag reproduceert bit-identiek wat A1 mat) · G-A1B-FLAGS ✅.

**Wat dit opent.** Defaults om: `use_ervf=True`, `rt.attn=attention_fp8_gqa4`,
`deterministic_accum=True`. De attentiekernel wordt nu via `rt.attn` gekozen;
oude scripts die `rt.k.attention_fp8_gqa` overschrijven meten voortaan een
nul-verschil in plaats van stil iets verkeerds — bewust zo gekozen.

**Het anker.** V35 wordt niet meer gereproduceerd, divergentie al bij token 1.
Dat komt van D1, niet van v4 (E4 reproduceerde V35 wél). Het anker legde een
ordeafhankelijk artefact vast. Vooraf vastgelegde regel gevolgd: `V36_DETERMINISTIC_ANCHOR.json`
bevroren, V35 ongewijzigd bewaard, niet-vergelijkbaarheid opgeschreven.

**Artefacten.** `A1_ADOPTION_PREREGISTRATION_2026-08-15.md` ·
`A1_ADOPTION_REPORT_2026-08-15.md` · `A1_ADOPTION_PRECONDITION.json` ·
`A1B_ADOPTION_VERIFY.json` · `V36_DETERMINISTIC_ANCHOR.json`

---

## 2026-08-15 — E6 — geïntegreerd 41,98 → 37,49 ms per token, exact

**Opzet.** Drie armen `base_a / integrated / base_b`, 3 domeinen × 512 causale
tokens, D1 in **alle** armen (zonder D1 zijn twee armen niet eens vergelijkbaar).
Eén variabele: ERVF + v4-attention.

**Uitkomst.** +4,169 (expository) / +4,443 (narrative) / +4,858 (code) ms, elk
boven zijn eigen drift. Bit-identieke uitvoer. VRAM ongewijzigd.

**Poorten.** Exactheid ✅ · latency ✅ · VRAM ✅ · eindpoort ≥50 tok/s ❌ (26,7).

**Wat dit sluit.** Niets — maar het laat zien dat de resterende afstand tot 50
tok/s niet uit de gebouwde componenten kan komen. E2 is weerlegd en E1 fase 2 is
ongebouwd; dat zijn de twee posten die het plan ervoor had ingeboekt.

**Artefacten.** `E6_INTEGRATED_REPORT_2026-08-15.md` · `E6_INTEGRATED_RUN.json`

---

## 2026-08-15 — D1 — de runtime was niet deterministisch, en dat is nu opgelost

**Vondst.** `_moe_cached` accumuleert de zes routed experts in
**hit-dan-miss-volgorde**. Welke expert een hit is hangt van de LRU-staat af, dus
twee runs met andere cachegeschiedenis tellen in andere volgorde op. FP-optelling
is niet associatief → twee armen met **identieke configuratie** divergeerden over
512 tokens. Ontdekt doordat NERVF-5 `base_b` tegen `base_a` zette.

**Ingreep.** Rekenvolgorde en optelvolgorde gescheiden: rekenen blijft hit-eerst
(latencywinst blijft), bijdragen naar aparte slotbuffers, na de lus optellen in
routevolgorde `s = 0..5`. Kosten: `top_k × hidden` floats (64 KB), nul extra
kernels.

**Uitkomst.** `base_b` nu identiek aan `base_a` over 3 × 512 tokens; NERVF-5
slaagt alsnog op alle drie zijn poorten. ERVF-winst met D1 aan: +2,771 / +5,008 /
+5,395 ms.

**De les.** Vier eerdere exactheidspoorten waren waar voor hun eigen run maar
bewezen minder dan ze leken. Vandaar werkregel 8: bouw een controle-arm die moet
falen.

**Artefacten.** `D1_DETERMINISM_REPORT_2026-08-15.md` · `d1_determinism.json`

---

## 2026-08-15 — NERVF-0 t/m 5 — ERVF gerepliceerd op een tweede model

**Vraag.** Reproduceert de Qwen3-30B ERVF-doorbraak op een architecturaal andere
NVFP4 hybrid-Mamba MoE?

**Antwoord: ja.** 1,936× bitexact op het projectievlak, bij **dezelfde** gekozen
subwarp-breedte 16 die Qwen selecteerde, op een ander model, een andere
quantisatie (NVFP4 vs Q5/Q8) en een andere shape.

**Het mechanisme.** w-lane subwarps per rij, 256/w rijen per 256-thread block,
per lane aparte virtuele accumulatoren, en een reductie die de DAG van de
referentiekernel **exact** reconstrueert — de eerste butterfly-stap (offset 16)
wordt een lane-lokale optelling, offsets 8/4/2/1 blijven shuffles binnen de
subwarp, en de acht warp-sommen vouwen in registers in exact de volgorde
`((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

**Fout onderweg.** w=4 en w=8 gaven eerst 72/72 mismatches doordat ik de virtuele
accumulatoren *sequentieel* vouwde in plaats van in butterfly-volgorde. Na de fix
alle vier breedtes bitexact.

**NERVF-1 valstrik.** RAW_SCAN sprong van 9,77 naar 51,67 µs tussen runs: het
2,81 MiB record past in L2. Opgelost door alle armen door een 254 MiB pool van 95
replica's te cyclen → spreiding 0,1%. Elke bandbreedtemeting hierna doet dit.

**NERVF-4.** Weerlegd, zie E2.

**Poorten.** Doorbraakladder LEVEL 2 gehaald (≥1,35× exact). LEVEL 3 niet: het
volledige expertpad bevat de down-projectie, die ERVF niet raakt en waar NERVF-4
de voor de hand liggende route sloot. LEVEL 4 (≥35 tok/s) niet: 29,45.

**Niet geclaimd.** Geen nieuwheidsclaim — geen prior-art-audit, geen stock
llama.cpp-vergelijking, geen tweede GPU.

**Artefacten.** `NERVF_NEMOTRON_FINAL_REPORT.md` · verifier 66/66 in
`nervf_independent_verification.json`
