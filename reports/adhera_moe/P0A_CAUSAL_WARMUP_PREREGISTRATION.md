# ADHERA-MoE P0A — causale 64-token-warmupcache

**Vastgelegd:** 2026-08-11, vóór berekening van deze policy.

## Transparantie

Deze exploratie gebruikt opnieuw de al geopende DHERA-validationroutes. De
vaste domeinbasisresultaten zijn bekend, inclusief hun code- en
instructionstaarten. Er wordt daarom geen bevestigende claim aan P0A ontleend.
Een positieve uitkomst opent uitsluitend een verse, blinde P0B.

## Enige policy

Per context van 1.024 tokens:

1. laad de reeds gelockte DCHERA-basis voor het aangeleverde domeinlabel;
2. verwerk tokens 0–63 in officiële token-, laag- en top-k-volgorde met de
   vaste 48-primary + 8-victimcache;
3. tel causaal alle officiële expert-ID's uit uitsluitend deze 64 tokens;
4. kies vóór token 64 exact 4.280 context-experts op aflopende warmup-count;
   ties volgen de reeds gelockte domein-trainingsrang en daarna laag/expert;
5. laad die volledige contextbasis, reset de 56 cache-slots en verwerk tokens
   64–1.023 zonder verdere aanpassing.

Er is geen lookahead, threshold-, warmuplengte-, cache- of rangsweep. Een expert
die niet in de eerste 64 tokens voorkomt, kan alleen via de vooraf gelockte
domeinrang in de contextbasis komen.

## Conservatieve verkeersboekhouding en gates

Aan token 0 én token 64 wordt elk een volledige geprojecteerde entropy-base van
4.280 experts toegerekend. Cold misses kosten daarnaast exact 9 MiB. Dezelfde
memory-, mean-, p95- en p99-gates als DCHERA gelden voor ieder domein; discrete
nearest-rank-percentielen. Onafhankelijke reproductie is verplicht.

Zelfs een positieve P0A bewijst geen werkelijke repacktijd, PCIe-overlap,
entropy-pack, modelkwaliteit, domeinclassificatie of tokens/s.
