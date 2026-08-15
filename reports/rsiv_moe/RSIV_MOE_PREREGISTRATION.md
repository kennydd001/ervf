# RSIV-MoE P0/P1 preregistratie

## Status en scheiding

- Hypothese: `RSIV_MOE_V1` / GhostWeights.
- Vastgelegd: `2026-08-11T07:28:07.2361674Z`, vóór de nieuwe rankcensus.
- CRAFT-MoE blijft onveranderd `closed_no_eureka`.
- Deze proef claimt geen Eureka, snelheid, kwaliteit of nieuwheid.
- DeepSeek-V4-Flash en Kimi K3 worden in P1 niet gedownload.

## Vraag

Kan een per-expert atlas met maximaal 32 inputrichtingen en 32
SwiGLU-intermediairrichtingen minstens 92% van toekomstige routed invocations
door beide subspacegates laten gaan en tegelijk minstens 10× minder
geprojecteerde koude expertbytes vragen dan een packed-int4 expertpad?

## Bevroren inputs

- Model: `deepseek-ai/DeepSeek-V2-Lite` Base.
- Modelrevision: `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`.
- Datasetrevision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Architectuur: hidden `d=2048`, routed SwiGLU-intermediate `m=1408`, 64
  routed experts, natuurlijke top-6, niet gerenormaliseerde routergewichten.
- Pilotlagen: 1, 13 en 26; layer 0 is dense en valt buiten de census.
- Blokken: 8×128 train, 4×128 validation en 4×128 test. Ieder blok is een
  onafhankelijke causale sequentie; er wordt niet over blokgrenzen gecompileerd.
- Causale prefix/future-split: posities `[0,96)` bouwen de basis; posities
  `[96,128)` zijn future en mogen de bevroren prefixbasis niet aanpassen.

De testactivaties mogen samen met de andere splits in één ongewijzigd raw
capturebestand worden vastgelegd. Teststatistieken worden pas berekend nadat
één globale validationkandidaat en diens SHA-256-record zijn geschreven.

## Representatie en numerieke conventies

Voor iedere expert wordt zonder centreren de oorsprongs-subruimte van de
gerouteerde rijen gebruikt. `Q` en `P` worden bepaald uit de rechter
singuliere vectoren van respectievelijk `X_e` en exact berekende `Z_e`.

- Rankcaps: `4, 8, 16, 32, 64, 128`.
- Primaire caps: alleen `4, 8, 16, 32`.
- Gedeelde gate-thresholds voor `rho_x` en `rho_z`:
  `0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10`.
- Residual ratio bij nulnorminput: nul als residual ook nul is, anders oneindig.
- Geen mean-centering, affine offset, testfit of experts-overstijgende basis.
- De opgeslagen BF16-matrices worden voor decompositie naar FP64 gepromoveerd.
- `stored_rank` gebruikt `max(n,d) * eps(float64) * sigma_max` als absolute
  singular-valuegrens.
- Effective rank is `exp(entropy(sigma²/sum(sigma²)))`; daarnaast worden de
  kleinste 90%, 95%, 99% en 99,9%-energieranks gerapporteerd.

## Twee vooraf vastgelegde evaluaties

1. **Offline calibration:** fit per expert uitsluitend op train; evalueer eerst
   validation en na kandidaatverzegeling eenmaal test.
2. **Causal prefix transfer:** fit per expert en per blok uitsluitend op de
   eerste 96 posities; evalueer de laatste 32 posities van hetzelfde blok.
   Blokken en future-invocations met een ontbrekende expertbasis tellen als
   miss, niet als ontbrekende data.

Een aparte causale onlinecurve verwerkt invocations in token-/slotvolgorde,
voegt bij een miss een DGKS-georthogonaliseerde residualrichting toe totdat de
cap is bereikt, en rapporteert ranktoevoegingen per token. Deze curve selecteert
geen kandidaat.

## Validationselectie

Eén globale `(rank_cap, threshold)` geldt voor `x` en `z` en voor alle drie
pilotlagen. Een kandidaat is validation-positief wanneer zowel offline als
causal-prefix, invocation-weighted over de drie lagen, voldoet aan:

```text
rank_cap <= 32
double_gate_fast_fraction >= 0.92
projected_routed_cold_byte_reduction >= 10.0x
```

Wanneer meerdere kandidaten slagen, wordt lexicografisch de kleinste
`threshold` en daarna de kleinste `rank_cap` gekozen. Wanneer geen kandidaat
slaagt, wordt uitsluitend één diagnostische kandidaat bevroren: maximale
`min(offline_fast/0.92, offline_reduction/10,
causal_fast/0.92, causal_reduction/10)`, met daarna lagere threshold en rank als
tie-break. Een diagnostische kandidaat kan de gefaalde validationgate niet
redden.

De selectie wordt met alle validationstatistieken en een SHA-256 van de raw
capture geschreven voordat de testvelden worden geopend.

## Koude-byteboekhouding

De optimistische packed-int4-referentie telt `0,5` byte per weightwaarde en
gelijke projectiegroottes:

- inputmiss: `G+U`, oftewel `2/3` van volledige routed expertbytes;
- intermediairmiss: `D`, oftewel `1/3`;
- beide hits: nul koude expertweightbytes.

Per invocation is de koude fractie dus
`(2*x_miss + z_miss)/3`; reductie is `1/mean(cold_fraction)`. Atlasreads,
basisprojecties, indices, quantisatie-error, aandacht, shared experts en
latency worden apart begrensd en niet als winst meegerekend. Dit is uitsluitend
een byteprojectie, geen runtimeclaim.

## Verplichte controles

Voor iedere laag en split:

1. `sum_e n_e == top_k * tokens`.
2. `rank(X_e) <= n_e` en `rank(Z_e) <= n_e`.
3. `sum_e[(d+2m)r_e + (m+d)s_e] <= (2d+3m)top_k*T`.
4. Routes uit het capturepad sluiten exact met de officiële router; maximale
   routergewichtfout is maximaal `1e-6`.
5. Volledige opgeslagen-rankbases reconstrueren `x`, `g`, `u`, `z` en `y`
   tegen directe FP32-operatoractie met relatieve L2 maximaal `2e-5` en maximale
   absolute fout maximaal `2e-4`.
6. Een 100%-fallbackcontrol is identiek aan het directe expertpad.

Een falende verplichte control maakt de run ongeldig, niet negatief.

## P1-besluit

- `screen_positive`: alle validationgates slagen en de eenmalig geopende test
  bevestigt ze voor dezelfde kandidaat.
- `screen_negative_v2`: validation of test faalt, terwijl alle controls slagen.
- `invalid`: minstens één verplichte control faalt.

Een V2-negatief screen sluit RSIV niet universeel; het rechtvaardigt evenmin
P2 op V2. Een hogere-E-replicatie vereist een apart registry-item en eigen
preregistratie. P2-operatorimages worden alleen gebouwd na `screen_positive`.

