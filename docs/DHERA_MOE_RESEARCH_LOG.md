# DHERA-MoE onderzoekslog

## 2026-08-11 — vaste budgetcache vooraf geregistreerd

HERA blijft gesloten. DHERA reserveert binnen 5,75 GiB exact 0,5 GiB voor 56
BF16-cacheslots en selecteert 4.280 entropy-experts op training-routerweight².
De policy heeft één primary slot per laag plus acht globale LRU-victimslots.
Nieuwe validationroutes uit vijf domeinen worden pas na input- en basislock
geopend. Er is geen cachegrootte-, selectie- of policiesweep toegestaan.

## 2026-08-11 — protocolverduidelijking vóór simulatie

De standaard victim-hit-transitie is expliciet gelockt: victim en primary van
de betreffende laag wisselen, zonder transfer. Percentielen gebruiken de
discrete nearest-rank-definitie. Er waren nog geen cache-events of
verkeerspercentielen berekend.

## 2026-08-11 — P0 negatief en onafhankelijk bevestigd

De vaste globale basis plus 48 primary- en 8 victimslots past met 5,749055 GiB
binnen het resident budget, maar faalt alle verkeersgates in alle vijf
domeinen. Gemiddelde H2D varieert van 91,336 MiB/token (general) tot
365,123 MiB/token (code), tegenover maximaal 64 MiB/token. Een afzonderlijke
implementatie reproduceert basis, routes, events, percentielen, geheugen en
gates: 22/22 controles slagen. P1 wordt niet geopend.

Dit sluit alleen de vaste globale basispolicy. Een domeingeconditioneerde basis
is niet door deze registry getest en vereist een afzonderlijke hypothese.
