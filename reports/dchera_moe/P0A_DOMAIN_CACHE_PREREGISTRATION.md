# DCHERA-MoE P0A — domeingeconditioneerde budgetcache

**Vastgelegd:** 2026-08-11, vóór berekening van een domeinspecifieke basis of
cache-uitkomst.

## Status van de data

De DHERA-validationroutes zijn al gebruikt voor de globale-basismeting. P0A is
daarom expliciet een exploratie op geopende routes. De domeinbasis zelf wordt
uitsluitend uit de overeenkomstige HERA-trainingroutes gekozen. Een positieve
P0A-uitkomst vereist een nieuwe, ongeopende P0B-bevestigingsset.

## Eén vaste policy

Voor ieder van de vijf bekende domeinlabels wordt afzonderlijk exact één basis
van 4.280 laag-expertparen gelockt. Selectie gebruikt binnen dat domein
aflopende trainingssom van `router_weight²`; ties: hogere count, lager
laagnummer, lager expert-ID.

Actief per context:

- 4.280 entropy-GPTQ-experts op de GPU;
- één exact BF16-primary-slot per laag en acht globale LRU-victimslots;
- 1.864 actieve cold-experts exact in host-RAM;
- dezelfde token-, laag-, top-k- en victim-swapvolgorde als DHERA;
- cache reset per 1.024 tokens;
- geen routewijziging, prefetch, lookahead of policiesweep.

Conservatief wordt aan token 0 van **iedere** context een volledige geprojecteerde
entropy-basewissel van 4.280 experts toegerekend. Dit modelleert het ongunstige
geval waarin opeenvolgende contexten steeds van domein wisselen.

## Gates

De memorygates blijven 5,75 GiB resident en 24 GiB actieve exact-cold host-RAM.
Per domein, inclusief basewissels, moet gemiddelde H2D maximaal 64 MiB/token,
p95 maximaal 144 en p99 maximaal 288 zijn. Percentielen gebruiken discrete
nearest rank. Alle event- en byteformules moeten onafhankelijk worden
gereproduceerd.

P0A positief opent alleen P0B op verse routes. P0A bewijst geen classifier,
pack, kwaliteit, werkelijke transferlatency, overlap of tokens/s.
