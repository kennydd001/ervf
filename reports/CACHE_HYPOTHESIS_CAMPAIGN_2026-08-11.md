# Cache-hypothesecampagne — eindrapport 2026-08-11

## Uitkomst

Er is **geen Eureka bewezen** en geen echte transferbenchmark geopend. Vier
vooraf vastgelegde policies zijn zonder gatewijziging getest; alle vier zijn
negatief. De memoryconstructie past steeds nét binnen het budget
(5,749055 GiB resident; 16,382812 GiB actieve exact-cold host-RAM), maar de
validationstaarten van vooral code en instruction blijven te zwaar.

De evidence omvat 163.840 nieuwe validationtokens, 48 lagen, top-k 8 en dus
62.914.560 officiële expert-ID's. De routecapture onderschepte iedere officiële
`topk`-call exact. De vier onafhankelijke verificaties leveren samen 77/77
geslaagde controles.

| Registry/policy | General | Code | Math | Multilingual | Instruction | Besluit |
|---|:---:|:---:|:---:|:---:|:---:|---|
| DHERA globale basis + primary/victim | FAIL | FAIL | FAIL | FAIL | FAIL | gesloten |
| DCHERA vaste domeinbasis | PASS | FAIL | PASS | PASS | FAIL | gesloten |
| ADHERA causale warmup-64 | PASS | FAIL | FAIL | PASS | FAIL | gesloten |
| LDHERA training-optimale laag-LRU | PASS | FAIL | PASS | PASS | FAIL | gesloten |

## Wat de data wel aantonen

1. **Geheugen is niet de directe blocker.** De 4.280-expertbasis, INT4-trunk en
   56 BF16-slots passen met ongeveer 0,000945 GiB marge.
2. **Domeinconditionering helpt sterk.** General daalt van 91,336 naar
   18,669 MiB/token gemiddeld; vergelijkbare verbeteringen treden in alle
   domeinen op.
3. **De doorslaggevende fout zit in bursts.** Code heeft bij de vaste
   domeinbasis een goed gemiddelde en p95 (24,570 en 63), maar p99 567.
   Instruction faalt p95/p99 met 225/666.
4. **Een korte causale warmup voorspelt de rest niet betrouwbaar.** Code
   verbetert, maar math en instruction verslechteren.
5. **Gemiddelde training-misses optimaliseren is niet genoeg.** De exacte
   laagallocatie houdt drie domeinen positief, maar code/instruction-staarten
   blijven falen.

## Claimgrens en volgende geldige richting

Deze campagne falsificeert vier concrete cachepolicies, niet iedere dynamische
cache. Een volgende geldige hypothese moet expliciet de **tail-risk** voorspellen
of begrenzen en mag niet opnieuw alleen gemiddelde hitrate optimaliseren. De
meest verdedigbare kandidaat is een training-only, per-context risicoscore die
vóór generatie een van meerdere vooraf verpakte domeinsubbases kiest, met een
verse blinde evaluatieset en een vooraf vastgelegde p99-objective. Die kandidaat
is hier niet getest en mag niet als Eureka worden beschreven.

## Primaire artefacten

- DHERA: `reports/dhera_moe/P0_FINAL_VERDICT.md`
- DCHERA: `reports/dchera_moe/P0A_FINAL_VERDICT.md`
- ADHERA: `reports/adhera_moe/P0A_FINAL_VERDICT.md`
- LDHERA: `reports/ldhera_moe/P0A_FINAL_VERDICT.md`
- Machineleesbare resultaten en onafhankelijke verificaties staan in dezelfde
  vier rapportmappen.
