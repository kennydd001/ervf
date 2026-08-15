# P1C — Qwen3-30B-A3B hogere-E rankscreen

**Vooraf geregistreerd:** 2026-08-11  
**Status bij registratie:** checkpointgewichten nog niet verworven; geen
validation- of testactivaties geopend.

## Hypothese

De schaalclaim van `RSIV_MOE_V1` voorspelt dat de grotere expertpopulatie van
Qwen3-30B-A3B de promptbelasting per expert verlaagt van gemiddeld 96 naar 64
invocaties. Daardoor moeten rank-32 input- en SwiGLU-intermediaire atlassen na
1.024 causale prefilltokens aanzienlijk vaker toekomstige routed activaties
opvangen dan op DeepSeek-V2-Lite.

## Bevroren inputs

- Model: `Qwen/Qwen3-30B-A3B-Base`.
- Revisie: `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
- Configuratie: `d=2048`, `m=768`, `E=128`, `top_k=8`, 48 lagen.
- Gewichten: de 16 officiële `model-*-of-00016.safetensors` BF16-shards,
  samen exact 61.066.575.648 bytes.
- Runtime-implementatie: `transformers==4.51.3`; de oudere projectpin 4.46.3
  bevat geen native `Qwen3MoeForCausalLM` en is vóór checkpointacquisitie
  vervangen. Bestaande V2-regressietests moeten daarna volledig slagen.
- Dataset: `Salesforce/wikitext`, configuratie `wikitext-2-raw-v1`, revisie
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Tokenisatie: Qwen-tokenizer op dezelfde vast gepinde ruwe WikiText-streams;
  geen hergebruik van V2-token-ID's.
- Lagen: `[0, 23, 47]` in Qwen zero-based indexing, na controle dat elk een
  routed MoE-laag is.

## Causaal protocol

- Twee vaste validationcontexten en twee vaste testcontexten.
- Iedere context bevat 1.152 tokens.
- Posities `[0:1024]` bouwen per expert uitsluitend de causale `Q_e`- en
  `P_e`-bases.
- Posities `[1024:1152]` zijn de enige future-evaluatie.
- Natuurlijke top-8 routerselectie en de officiële genormaliseerde
  routergewichten blijven exact behouden.
- Validation selecteert precies één globale `(rank_cap, threshold)`; test
  wordt daarna één keer geopend.

## Grid en metingen

```text
rank_cap: 4, 8, 16, 32, 64, 128
threshold: 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10
```

Per laag/expert/context worden vastgelegd:

- prefixobservaties, exacte `x`-rank en exacte `z`-rank;
- future-residualratio's voor input `x` en exact SwiGLU-intermediair `z`;
- x-fast, z-fast en double-fast per rank/threshold;
- gewogen p50/p95/p99 en rare-expertdekking;
- analytische packed-int4 koude bytes voor G/U/D-misses;
- rankgroei en benutting van de `kT/E`-bound.

## Controles

Verplicht vóór een validation-lock:

1. model-ID, commit, configuratie, shardnamen en totale shardbytes sluiten;
2. tokenizer- en datasetrevisie sluiten;
3. routerexpert-ID's en routergewichten sluiten met de officiële forward;
4. `sum(expert_counts) = tokens * top_k` per laag/context;
5. gecentreerde noch niet-causale activaties mogen in een prefixbasis lekken;
6. full-rank FP64-projectie van opgeslagen `x/z` reconstrueert met relatieve
   L2-fout maximaal `2e-12`;
7. alle activaties, routes, ranks en residuals zijn eindig en binnen bereik;
8. testslices blijven cryptografisch gesloten tot de validation-lock bestaat.

Een aanvullende BF16-bitexactheidstest tussen verschillende GEMM-batchvormen
is diagnostiek, geen verplichte controle; P1A heeft aangetoond waarom die eis
numeriek ongeldig is. FP32-operatoridentiteit blijft vereist wanneer echte
operatorimages worden gebouwd, maar P1C is een activatie-rankscreen.

## Selectieregel

Kies op validation eerst uit kandidaten met `rank_cap <= 32` die beide primaire
gates halen. Bij meerdere kandidaten wint achtereenvolgens: lagere rank, lagere
threshold, hogere double-fastfractie en hogere koude-bytereductie. Als geen
kandidaat slaagt, vergrendel de kandidaat met de hoogste minimum-gatefractie;
die blijft expliciet diagnostisch en kan P1C niet positief maken.

Primaire P1C-gates:

```text
rank_cap <= 32
double_gate_fast_fraction >= 0.92
projected_routed_cold_byte_reduction >= 10.0x
```

## Vooraf vastgelegde uitkomsten

- **P1C positief:** de vergrendelde kandidaat haalt beide gates op validation
  én test. Alleen een nieuwe P2-preregistratie is toegestaan.
- **P1C negatief / hard falsified:** rank 32 haalt op een geldige,
  representatieve prefilltest minder dan 80% double-fast. Omdat dit reeds op V2
  is gebeurd, sluit `RSIV_MOE_V1` als `falsified_rank_working_set`.
- **Inconclusive:** uitsluitend bij een onherstelbare integriteits-, resource-
  of modelcompatibiliteitsfout; numeriek tegenvallende data is geen reden om
  inconclusive te kiezen.

Geen P1C-uitkomst is op zichzelf een kwaliteits-, runtime- of Eureka-bewijs.

## Resource-en acquisitiegrenzen

- Downloadpad: `models/qwen3-30b-a3b-base`.
- Alleen exacte revision en noodzakelijke config/tokenizer/index/shardbestanden.
- Minimaal 90 GiB vrij vóór download.
- Proces-RSS tijdens capture maximaal 32 GiB; GPU-geheugen maximaal 7,5 GiB.
- Bij afgebroken acquisitie wordt hervat; een gedeeltelijke checkpointmap is
  geen experimenteel resultaat.
