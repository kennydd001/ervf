# Vooraf vastgelegde mass-budgetconfirmatie

Vastgelegd vóór uitvoering op 2026-08-10. Dit bestand mag na de run alleen met
een gedateerde addendumsectie worden uitgebreid; de onderstaande policy's en
gates mogen niet worden aangepast op basis van de uitkomst.

## Onaangeraakt evaluatievenster

- model: `deepseek-ai/DeepSeek-V2-Lite` Base, commit
  `604d5664dddd88a0433dbae533b7fe9472482de0`;
- dataset: gepinde WikiText-2-raw-v1 validatie en test, commit
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- tokenoffset: 4.096 IDs onafhankelijk vanaf het begin van iedere split;
- omvang: 16 blokken van 128 tokens per split, dus 2.048 validatie- en 2.048
  testtokens;
- cache: onafhankelijk lege LRU met capaciteit 32 per blok en MoE-laag;
- alle beleidskeuzes hieronder zijn vóór inspectie van dit venster vastgezet.

## Vooraf vastgelegde policy's

- exacte `original`-control;
- oude baseline `max_rank:j5:7`;
- vaste Cache-Prior `cache_prior:j2:0.0275` en
  `cache_prior:j2:0.095`;
- kandidaat `mass_budget:j2:0.004` en `mass_budget:j2:0.018`.

De lage policy's vormen het kwaliteitsgerichte paar; de hoge policy's het
loadgerichte paar. Er wordt na deze run geen nieuwe δ of λ gekozen om de gate
alsnog te halen.

## Succesgates

De confirmatie is geslaagd wanneer alle controles slagen en de inhoudelijke
gates als volgt uitvallen:

1. `original` heeft op validatie én test KL=0, CE-delta=0, top-1=1 en maximale
   laagfout ≤ `1e-6`.
2. `mass_budget:j2:0.004` heeft op de testset zowel meer loadreductie als lagere
   teacher→candidate-KL dan `max_rank:j5:7`.
3. Voor minstens één vooraf gekoppeld mass-budget/Cache-Prior-paar geldt op
   zowel validatie als test: de mass-budget-loadreductie ligt maximaal 2,0
   procentpunt lager en de KL is minstens 5% lager.
4. Voor `mass_budget:j2:0.004` is de relatieve test-CE-puntschatting ≤ `+0,5%`
   en de bovengrens van het 95%-blokbootstrapinterval ≤ `+1,0%`.
5. Alle zestien testblokken van `mass_budget:j2:0.004` hebben positieve
   loadreductie.

Een mislukte gate blijft als falsificatie staan en wordt niet door een nieuw
afgesteld punt vervangen.

## Uitvoercommando

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_modelwide_cache_routing_pareto.py `
  --blocks-per-split 16 --block-size 128 --token-offset 4096 --capacity 32 `
  --corpus-preset wikitext `
  --policy original --policy max_rank:j5:7 `
  --policy cache_prior:j2:0.0275 --policy mass_budget:j2:0.004 `
  --policy cache_prior:j2:0.095 --policy mass_budget:j2:0.018 `
  --report-name preregistered_wikitext_offset4096_mass_budget_confirmation.json
```

## Addendum na uitvoering — 2026-08-10

De run is zonder parameterwijziging voltooid. Bronrapport:
`baseline/preregistered_wikitext_offset4096_mass_budget_confirmation.json`.

| Gate | Resultaat | Oordeel |
|---|---|---|
| 1. exacte control | KL `0`, CE `0`, top-1 `1`, maximale laagfout `0` | **geslaagd** |
| 2. δ=0,004 versus rank-7 op test | load `14,017%` versus `10,559%`; KL `0,003704` versus `0,004389` | **geslaagd** |
| 3. gekoppeld paar op beide splits | δ=0,004 versus λ=0,0275: 1,379/1,393 pp minder load maar 15,54%/15,11% lagere KL op validatie/test | **geslaagd** |
| 4. CE-sanitygate | testpunt `−0,057%`; 95%-CI `−0,171%–+0,060%` | **geslaagd** |
| 5. ieder testblok bespaart | 16 van 16 positieve loadreductie | **geslaagd** |

Aanvullend heeft δ=0,018 op test `40,271%` loadreductie bij KL `0,010137`,
tegen `42,195%` en KL `0,011255` voor λ=0,095. Dit hoge paar mist op
validatie de vooraf gestelde loadtolerantie met 0,274 procentpunt, maar gate 3
vereiste expliciet slechts één van de twee vooraf gekoppelde paren; het lage
paar voldoet op beide splits ruim.
