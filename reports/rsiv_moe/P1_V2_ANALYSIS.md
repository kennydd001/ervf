# GhostWeights / RSIV-MoE — P1 V2-analyse

## Conclusie

De DeepSeek-V2-Lite-pilot is een duidelijke **`screen_negative_v2`**. Alle
algebraïsche en capturecontroles slagen, maar de bezochte expertlokale
activaties vormen op deze workload geen klein, overdraagbaar rank-32 working
set. Dit blokkeert P2 op V2. Het is nog geen universele falsificatie van RSIV,
omdat de vooraf vastgelegde schaalhypothese afzonderlijk op een hogere-E-familie
moet worden getest.

## Primaire kandidaat

Validation vond geen enkele positieve kandidaat. Conform de preregistratie is
daarom uitsluitend de lexicografische diagnostische kandidaat verzegeld:

```text
rank_cap = 4
rho_x = rho_z = 0.001
selection_kind = diagnostic_validation_failure
```

Die kandidaat behaalt op zowel validation als de eenmalig geopende test:

| Evaluatie | Validation double-fast | Validation cold reduction | Test double-fast | Test cold reduction |
|---|---:|---:|---:|---:|
| Offline trainbasis | 0,000% | 1,000× | 0,000% | 1,000× |
| Causale prefix 96→future 32 | 0,000% | 1,000× | 0,000% | 1,000× |

De gate was voor beide evaluaties `>=92%` double-fast én `>=10×` minder
geprojecteerde koude expertbytes.

## Sterkste toegestane diagnostiek

Zelfs nadat de kandidaat al was verzegeld, levert de ruimste gridcombinatie
`rank=128, threshold=0,10` nauwelijks een fast path:

| Split | Offline double-fast | Offline cold reduction | Causaal double-fast | Causale cold reduction |
|---|---:|---:|---:|---:|
| Validation | 0,629% | 1,006× | 0,000% | 1,000× |
| Test | 0,749% | 1,008× | 0,000% | 1,000× |

Bij de primaire limiet `rank=32, threshold=0,10` is test slechts `0,412%`
double-fast en `1,006×` cold reduction; de causale transfer blijft nul.

Per laag op test bij `rank=128, threshold=0,10`:

| Laag | Offline double-fast | Offline cold reduction | Causaal double-fast |
|---:|---:|---:|---:|
| 1 | 0,391% | 1,004× | 0,000% |
| 13 | 1,465% | 1,015× | 0,000% |
| 26 | 0,391% | 1,004× | 0,000% |

Dit sluit uit dat alleen één ongelukkige pilotlaag het aggregate resultaat
domineert.

## Waarom de exacte schaalwet hier niet helpt

De identiteit

```text
S_layer <= (2d + 3m) * top_k * T
```

blijft exact. Het probleem is dat de bovengrens vrijwel volledig wordt
verzadigd: voor bijna iedere expert zijn de opgeslagen `x`- en `z`-rijen
lineair onafhankelijk. De full-rank atlas gebruikt `99,902–100,000%` van de
expert-count-cancellationbound.

Invocation-weighted stored ranks:

| Split | Laag | Input rank p50 / p95 | Intermediate rank p50 / p95 |
|---|---:|---:|---:|
| Train | 1 | 101 / 163 | 101 / 163 |
| Train | 13 | 137 / 610 | 137 / 610 |
| Train | 26 | 104 / 215 | 104 / 215 |
| Test | 1 | 51 / 119 | 51 / 119 |
| Test | 13 | 75 / 263 | 75 / 263 |
| Test | 26 | 55 / 174 | 55 / 174 |

De lagere entropy-effective-rank redt de gate niet. Op train is de gemiddelde
input-effective-rank bijvoorbeeld `47,83`, `55,96` en `18,03` voor lagen
1/13/26, maar de gemiddelde 99%-energierank is respectievelijk `86,88`,
`175,91` en `98,03`. Voor `z` zijn die 99%-energieranks `89,88`, `158,18` en
`99,00`. Een lage spectral-entropywaarde betekent hier dus niet dat een
rank-32-subspace toekomstige vectoren binnen 10% residual opvangt.

## Causale rankgroei

Bij de bevroren threshold `0,001` en diagnostische cap 128 voegt iedere routed
invocation in de vier testblokken een nieuwe richting toe:

| Laag | Inputtoevoegingen | `z`-toevoegingen | Double-fast | Cold reduction |
|---:|---:|---:|---:|---:|
| 1 | 3.072 | 3.072 | 0,000% | 1,000× |
| 13 | 3.072 | 3.072 | 0,000% | 1,000× |
| 26 | 3.072 | 3.072 | 0,000% | 1,000× |

Dat is exact zes nieuwe input- én zes nieuwe intermediairrichtingen per token
per laag. Binnen de gemeten 128-tokenblokken is er geen teken van saturatie.
Dit is sterke V2-tegenwind, maar nog geen lang-decodebewijs.

## Controles

- Alle officiële route-ID's sluiten bit-exact; maximale routergewichtfout is
  nul.
- Iedere laag heeft exact `2.048 × 6 = 12.288` captured invocations.
- `rank(X_e) <= n_e`, `rank(Z_e) <= n_e`, countidentiteit en de opslagbound
  slagen voor train, validation en test.
- Full-rank `x/g/u/z/y`-operatorimages slagen per laag en globaal binnen
  relatieve L2 `2e-5` en maximale absolute fout `2e-4`.
- De raw capture, validation-lock en testresultaten zijn gehasht en append-only.

De diagnostische BF16-`z`-herberekening is niet bit-exact wanneer dezelfde
rijen naar andere GEMM-batchvormen worden gegroepeerd. Addendum 001 legt vast
waarom dit geen geldige operator- of capturefout is; alle geregistreerde FP32-
identiteitscontroles slagen.

## Claimgrens en volgende beslissing

Deze proef meet expertlokale teacher-state-rank en geprojecteerde koude bytes op
128-token WikiText-blokken. Zij meet geen CE, rolloutkwaliteit, packed runtime,
SSD-stalls of 512-token decode. Daarom volgt geen snelheidsclaim en geen
Eureka-verdict.

P2 op V2 blijft gesloten. De enige preregistratie-conforme vervolgproef is een
nieuwe P1-rankcensus op één hogere-E MoE-familie. Alleen wanneer die de
rank-32/page-faultscreen wél haalt, mag een echte operatorimage-oracle worden
gebouwd.

