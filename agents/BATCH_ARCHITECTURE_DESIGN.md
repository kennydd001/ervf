# Batch>1 architecture design — what the integration would actually need

Datum: 2026-08-16 · status: ontwerp, geen code · vereist door `TODO.md`'s
eigen volgende-stap-aanwijzing ("architectuurontwerp... geen kernels")

## Waarom dit bestaat

`agents/RESEARCH_NOTEBOOK.md` (2026-08-16, meerdere blokken) bewijst het
kernmechanisme van batch>1 — expert-fetch delen over sequenties — fysiek
correct en sneller, voor zowel up_proj (1,71× opgeteld over alle 23 lagen)
als down_proj (1,91×, unie-van-sparsity-maskers, correcte generalisatie,
niet simpelweg gekopieerd van up_proj). Beide zijn geïsoleerde
componentmetingen, geen runtime-integratie. Dit document is de brug ertussen:
wat zou een echte batch>1-runtime nodig hebben, wat is het grootste risico,
en waarom is dat een meerdere-weken-taak in plaats van iets om nu snel te
bouwen.

**Dit is geen bouwplan om blind te volgen — het is de eerlijke inventarisatie
die nodig is vóór iemand daaraan begint, geschreven zodat de volgende agent
niet bij nul hoeft te beginnen met nadenken.**

## Wat een batch>1-stap zou moeten doen, stap voor stap

1. **N sequenties, elk hun eigen positie/lengte.** In tegenstelling tot deze
   sessie's prototypes (die N sequenties op *dezelfde* stap-index vergelijken)
   heeft een echte serving-runtime sequenties op *verschillende* posities
   (continuous batching: sequentie A genereert token 40, sequentie B token 5,
   tegelijk in dezelfde stap). Dat betekent: geen enkele buffer kan een simpele
   `[hidden]`-vorm houden — alles wordt `[N, hidden]`, en KV-cache/SSM-state
   wordt `[N, max_ctx, ...]` / `[N, ...]` in plaats van `[max_ctx, ...]` /
   `[...]`.
2. **Per-sequentie routing, dan de unie bepalen.** `_route_device` zou N keer
   moeten draaien (of één keer gebatcht, zelfde soort GEMV-batching als deze
   sessie al voor up_proj/down_proj deed) om N×top_k route-ids te krijgen,
   dan de unie berekenen — exact het mechanisme dat
   `diag_cross_sequence_union.py` en de twee `proto_batch_*`-scripts al
   losstaand bewezen hebben, nu ingebed in de stap-lus zelf.
3. **Cache-toewijzing tegen de unie, niet per sequentie.** De bestaande
   device-LRU (`cache_assign`, `alloc_device_cache`) is per-laag, niet
   per-sequentie — dat werkt in principe al goed samen met een unie-aanpak
   (één cachetoewijzing per laag per stap, gevoed door de unie-ids in plaats
   van één sequentie se ids). Vereist wel: `cache_assign`'s kernel aanpassen
   om N×top_k ids te ontvangen en te dedupliceren in plaats van top_k.
4. **up_proj: gedeelde fetch, per-sequentie GEMV.** Al bewezen
   (`proto_batch_moe_multilayer.py`). Elke sequentie se GEMV leest van de
   gedeelde cache-slot, met haar eigen `normed`-rij.
5. **down_proj: unie-van-maskers fetch, per-sequentie gemaskeerde som.** Al
   bewezen (`proto_batch_down_proj.py`). Vereist per-laag een unie-
   maskerberekening (OR over de sequenties die een gegeven expert kozen) —
   een extra kleine kernel, niet gebouwd.
6. **Shared expert: triviaal — nu ook fysiek bevestigd (2026-08-16), in
   tegenstelling tot de Mamba-aanname die fout bleek.** Draait toch al voor
   elke stap ongeacht routing — wordt gewoon `[N, hidden]` in plaats van
   `[hidden]`, geen nieuwe deel-logica nodig (het gewicht was al niet
   expert-afhankelijk). `pro_research/diag_shared_expert_n_scaling.py`: N
   sequentiële aanroepen van de bestaande GEMV tegen N echte activaties,
   N∈{1,2,4,8,16} — ms/sequentie blijft vlak (0,0378→0,0345 ms), verhouding
   tegen ideaal-lineair 0,85-0,92 (iets efficiënter dan lineair). Zelfde
   discipline als de Mamba-check, dit keer bevestigt de meting de
   oorspronkelijke aanname in plaats van hem te corrigeren.
7. **Attentie, Mamba, KV-cache: GEEN deel-mogelijkheid van deze soort — en
   voor Mamba specifiek, fysiek gemeten in plaats van bij analogie
   aangenomen (correctie 2026-08-16).** Attentie- en Mamba-gewichten zijn
   **niet expert-geselecteerd** — elke sequentie gebruikt toch al dezelfde
   Q/K/V/O-, Mamba-in/out-gewichten. Er is dus niets te dedupliceren. Voor
   attentie is dat gemeten: `diag_attention_n_scaling.py` — 94-97% van
   ideaal lineair, ms/sequentie nagenoeg vlak (0,091-0,096 ms) over
   N∈{1,2,4,8,16}. **Voor Mamba bleek de eerste versie van dit document het
   mis te hebben** door aan te nemen dat dezelfde conclusie overdraagt:
   `diag_mamba_n_scaling.py` (Mamba se `in_proj`, een fysiek andere
   FP8-per-tensor-kernel dan attentie se BF16-ERVF-pad) mat **mild
   supra-lineaire** schaling, niet lineair — ms/sequentie **stijgt** van
   0,177 ms (N=1) naar ~0,203-0,206 ms (N=8-16), een reële ~15% straf bij
   grotere N, vermoedelijk gerelateerd aan de grotere rijenvorm
   (rows=10304) en resourcecontentie bij herhaalde back-to-back launches
   (oorzaak niet verder onderzocht). **Dit betekent dat de aggregate-
   doorvoerwinst van batch>1 kleiner zal zijn dan de MoE-alleen-cijfers
   suggereren, én iets kleiner dan de eerdere (te optimistische) rekensom
   hieronder aannam** — MoE is 57,8% van het token (componentafbraak),
   de rest (attentie 14,9% vlak, Mamba ~0% maar mild duurder bij N,
   lm_head+shared 10,1%, overig) profiteert niet op dezelfde manier, en
   Mamba wordt zelfs iets duurder per sequentie.
8. **CUDA-graph-implicaties.** De huidige graph is gebouwd voor exact één
   token, één sequentie, vaste posities via `self._pos_dev`. Een batch>1-
   graph zou vaste `N_max`-grootte buffers nodig hebben (padding voor
   inactieve slots wanneer een sequentie klaar is en een nieuwe nog niet is
   ingestapt) — analoog aan hoe productie-LLM-serving-systemen "continuous
   batching" met een actief-masker oplossen. Herbouwen van de graph bij elke
   sequentie-wissel zou de graph-residentiewinst van V4-V6 weer opeten; een
   masker-gebaseerde aanpak (altijd N_max slots, sommige inactief) behoudt
   die winst maar kost compute op inactieve slots.

## Grootste onbekende risico's, in volgorde van belang

1. **Attentie/Mamba/KV-cache verdunnen de aggregate winst — en meer dan de
   eerste versie van dit document aannam.** Zoals hierboven: 57,8% van het
   token (MoE) kan profiteren, de rest niet op dezelfde manier. Een grove,
   expliciet-gelabelde bovengrens (geen meting): als MoE's kost naar ~0 zou
   gaan door perfecte deling maar de rest ongewijzigd blijft (aanname, niet
   gemeten), zou dat token van ~21 ms naar ruw ~21×0,42 ≈ 8,8 ms per
   **sequentie-equivalent** kunnen — een aggregate ~114 tok/s over de hele
   batch, **niet** 114 tok/s per sequentie. Dit is aritmetiek op een aanname
   (perfecte MoE-deling, nul overhead, "rest ongewijzigd"), geen
   voorspelling — precies het soort rekensom dat de S10-preregistratie ook
   deed vóór de MTP-meting, en die bleek daar te optimistisch. **Inmiddels
   bevestigd deels te optimistisch hier ook:** attentie blijft inderdaad
   ~ongewijzigd (gemeten), maar Mamba wordt mild duurder per sequentie bij
   grotere N (gemeten, ~15% straf bij N=8-16) — dus "de rest ongewijzigd"
   klopt niet volledig, en de werkelijke aggregate bovengrens ligt iets
   lager dan 114 tok/s. Geen nieuw getal herberekend (zou weer aanname-op-
   aanname zijn); behandel 114 als een ruwe, nu bevestigd-optimistische
   bovengrens.
2. **Compute schaalt wél met N, én de twee deel-mechanismen samen zijn
   minder dan de som der delen (bijgewerkt 2026-08-16).**
   `proto_batch_moe_multilayer.py` liet zien dat GEMV-compute-tijd vlak
   blijft tussen naive/batched per PAAR in isolatie — maar
   `proto_batch_moe_layer_combined.py` (up_proj- én down_proj-deling
   **tegelijk** in één laag, N=8, eerste zo'n meting) vond dat down_proj se
   `down_masked`-GEMV zelf **langzamer** wordt in de gecombineerde meting
   (7,037 ms) ondanks gelijke FLOP's per sequentie — vermoedelijk
   geheugenlocaliteit die verslechtert door de grotere unie-mirror. Netto
   bleef de gecombineerde winst positief en bitexact (**1,209×, +20,9%**,
   0/48 mismatches) maar **kleiner dan de afzonderlijke up_proj/down_proj-
   cijfers deden vermoeden**. Les: de twee mechanismen apart bewijzen
   volstaat niet om hun gecombineerde winst te voorspellen — moet samen
   gemeten worden. Bij grote N kan compute (en dit locatie-effect) op een
   gegeven moment de fetch-winst verder inhalen (roofline-plafond blijft
   gelden, nu voor compute i.p.v. PCIe).
3. **Continuous batching versus deze sessie se "alle N tegelijk op dezelfde
   stap"-aanname — gemeten, risico grotendeels gesloten (2026-08-16).**
   `pro_research/diag_staggered_position_union.py`: N=4, vaste offsets
   0/7/15/23 (elke sequentie op haar eigen echte generatiediepte) versus
   lockstep, zelfde onderliggende trajectdata. Unie **89,4% (lockstep) vs
   91,4% (staggered)** van max — slechts +1,9 procentpunt groter, geen
   ineenstorting. Continuous batching vernietigt het routing-overlap-deel-
   potentieel dus niet, verzwakt het licht. **Nog niet gemeten:** of dit ook
   geldt bij grotere spreiding (N=8/16, offsets tot honderden stappen) en de
   volledige runtime-integratie zelf (routing-unie ingebed in een staplus die
   N onafhankelijke posities bijhoudt) blijft ongebouwd.
4. **VRAM — gemeten, en het herkadreert het risico (2026-08-16).**
   `pro_research/diag_batch_vram_cost.py`: exacte kost per extra sequentie,
   nagerekend uit `runtime.py`'s eigen `_alloc_state`-formules: **60,16 MiB**
   (Mamba ssm+conv-state 48,2 MiB + KV-cache 12,0 MiB — Mamba domineert,
   want maar 6 van 52 lagen zijn volledige attentie). Bij het eager+
   device-cache-bedrijfspunt (geen graph-capture) is er **1.771 MiB vrij**,
   echt gemeten — ruimte voor **29 extra sequenties (N tot 30)** zonder iets
   te verlagen. Bij volledige V6-graph-capture blijft de eerder gemeten
   **0 MiB vrij** staan (V4-preregistratie, niet opnieuw gemeten hier).
   **Het probleem is dus niet batch>1's eigen VRAM-kost** (60 MiB/sequentie
   is klein) **maar dat graph-capture zelf al het budget opeet** vóór
   batch>1 er iets bij vraagt. Een eager (niet-graph-resident) batch>1-
   integratie heeft ruim budget; een graph-resident integratie (V4-V6's
   eigen winst) zou eerst de graph-capture-kost moeten verlagen
   (`contexts_max` of cache-capaciteit omlaag) vóór er N=2 bij kan — een
   reële afweging tussen twee al bewezen hefbomen, nu voor het eerst
   gekwantificeerd.
5. **Wat "tok/s" betekent verandert.** Dit hele onderzoeksdoel (100 tok/s) is
   impliciet single-stream geweest. Batch>1's winst is **aggregate**
   doorvoer, niet per-sequentie-latentie — die kan zelfs iets slechter worden
   (meer werk per stap, dus elke individuele sequentie wacht iets langer per
   token). Of dat "telt" voor het gestelde doel is een keuze die de
   gebruiker moet maken, geen technisch feit.

## Wat NIET gedaan is, met opzet

Geen regel productiecode voor de volledige integratie. Elk stuk hierboven dat
al bewezen is (up_proj-deling, down_proj-deling) staat als losstaand,
bitexact geverifieerd prototype in `pro_research/`. De rest (routing-unie
ingebed in de staplus, cache_assign aangepast voor N×top_k, batch-dimensie op
alle buffers, graph met actief-masker, attentie/Mamba batch-vorm) is
ontwerp, geen code — precies zoals `TODO.md` als volgende stap aangaf.

## Aanbeveling voor wie dit oppakt

Begin niet met de volledige integratie. Begin met risico 1 hierboven fysiek
meten: bouw een klein, geïsoleerd prototype dat de **attentie**-kost voor
N=2 sequenties batcht (gewoon N× dezelfde GEMV met een batch-dimensie, geen
deel-mechanisme nodig omdat er niets te delen valt) en meet of dat werkelijk
~lineair schaalt zoals aangenomen. Als dat klopt, is de grove bovengrens in
risico 1 een redelijk uitgangspunt voor een preregistratie met een harde
poort; zo niet, dan moet de hele rekensom over.

**Gedaan, zelfde dag.** `pro_research/diag_attention_n_scaling.py`: de
bestaande Q-projectie-GEMV N keer gedraaid tegen N echte activaties,
N ∈ {1,2,4,8,16}. Resultaat: 94-97% van het ideale lineaire model, ms/
sequentie nagenoeg constant (0,091-0,096 ms) over de hele reeks. **Bevestigt
de aanname: attentie schaalt ~lineair, geen launch-overhead-speling zoals
MoE had.** De grove ~114 tok/s-bovengrensrekensom hierboven gebruikte precies
deze aanname (attentie/Mamba/rest ongewijzigd) — die aanname staat nu
gestaafd, niet weerlegd. Zie `agents/RESEARCH_NOTEBOOK.md` 2026-08-16.
