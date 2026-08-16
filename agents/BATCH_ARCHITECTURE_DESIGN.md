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
6. **Shared expert: triviaal.** Draait toch al voor elke stap ongeacht
   routing — wordt gewoon `[N, hidden]` in plaats van `[hidden]`, geen
   nieuwe deel-logica nodig (het gewicht was al niet expert-afhankelijk).
7. **Attentie, Mamba, KV-cache: GEEN deel-mogelijkheid van deze soort.**
   Dit is de belangrijkste eerlijke waarschuwing van dit document. Attentie-
   en Mamba-gewichten zijn **niet expert-geselecteerd** — elke sequentie
   gebruikt toch al dezelfde Q/K/V/O-, Mamba-in/out-gewichten. Er is dus
   niets te dedupliceren; de kost schaalt al ~lineair met N (N keer zoveel
   GEMV-werk, weliswaar misschien met launch-batching zoals deze sessie al
   voor MoE deed, maar zonder het PCIe-amortisatie-effect dat MoE's winst
   gaf). **Dit betekent dat de aggregate-doorvoerwinst van batch>1 kleiner
   zal zijn dan de MoE-alleen-cijfers suggereren** — MoE is 57,8% van het
   token (componentafbraak, eerder vandaag), de rest (attentie 14,9%, Mamba
   ~0%, lm_head+shared 10,1%, overig) profiteert niet op dezelfde manier.
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

1. **Attentie/Mamba/KV-cache verdunnen de aggregate winst.** Zoals hierboven:
   57,8% van het token (MoE) kan profiteren, de rest niet op dezelfde manier.
   Een grove, expliciet-gelabelde bovengrens (geen meting): als MoE's kost
   naar ~0 zou gaan door perfecte deling maar de rest ongewijzigd blijft
   (aanname, niet gemeten), zou dat token van ~21 ms naar ruw ~21×0,42 ≈
   8,8 ms per **sequentie-equivalent** kunnen — een aggregate ~114 tok/s
   over de hele batch, **niet** 114 tok/s per sequentie. Dit is aritmetiek
   op een aanname (perfecte MoE-deling, nul overhead), geen voorspelling —
   precies het soort rekensom dat de S10-preregistratie ook deed vóór de
   MTP-meting, en die bleek daar te optimistisch. Behandel dit getal zo.
2. **Compute schaalt wél met N (al gemeten, geen straf, maar ook geen
   winst).** `proto_batch_moe_multilayer.py` liet zien dat GEMV-compute-tijd
   vlak blijft tussen naive/batched per PAAR — maar er zijn nog steeds N×
   top_k paren te berekenen, dus totale compute schaalt met N ongeacht
   deling. Bij grote N kan compute op een gegeven moment de fetch-winst
   inhalen (roofline-plafond blijft gelden, nu voor compute i.p.v. PCIe).
3. **Continuous batching versus deze sessie se "alle N tegelijk op dezelfde
   stap"-aanname.** De prototypes vergeleken N sequenties op identieke
   stap-index. Een echte serving-runtime heeft sequenties op willekeurige,
   ongelijke posities — de route-unie-berekening blijft hetzelfde mechanisme,
   maar is nooit gemeten onder die realistischere voorwaarde.
4. **VRAM.** N-voudige KV-cache/SSM-state kost VRAM die er op deze 8 GiB-kaart
   al niet is (0 MiB vrij tijdens V6). Grote N (16) is voor lange contexten
   waarschijnlijk niet haalbaar zonder de cache-capaciteit fors te verlagen —
   een nieuwe afweging, niet gemeten.
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
