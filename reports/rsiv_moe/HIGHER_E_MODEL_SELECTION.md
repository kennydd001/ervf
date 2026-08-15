# RSIV-MoE hogere-E modelselectie

**Datum:** 2026-08-11  
**Besluit:** `Qwen/Qwen3-30B-A3B-Base` is de eerste hogere-E-replicatie.

## Doel

P1A en P1B op DeepSeek-V2-Lite zijn beide negatief. P1B sluit bovendien uit
dat alleen de korte 96-tokenprefix de oorzaak was. De vooraf vastgelegde
RSIV-schaalvoorspelling vereist nu een tweede architectuur met meer experts.
Deze keuze opent geen Qwen-testdata en verandert geen bestaande gate.

## Officiële checkpointfeiten

| Eigenschap | Qwen3-30B-A3B Base | Kimi-Linear-48B-A3B Base |
|---|---:|---:|
| Gepinde revisie | `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` | `3b171c17bfc4ee348599b6781a2ca8715c21c8dc` |
| Repositoryopslag | 61.066.575.648 B (56,873 GiB) | 98.270.961.861 B (91,522 GiB) |
| Hidden size `d` | 2.048 | 2.304 |
| MoE-intermediate `m` | 768 | 1.024 |
| Experts `E` | 128 | 256 |
| Top-k | 8 | 8 |
| Lagen | 48 | 27 |

Bronnen: de officiële
[Qwen-revisie](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/tree/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9),
[Qwen-configuratie](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9/config.json) en de officiële
[Kimi-repository](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/tree/3b171c17bfc4ee348599b6781a2ca8715c21c8dc).

## Selectiereden

Voor een prompt van 1.024 tokens is de verwachte routerbelasting
`top_k * T / E`:

| Model | Gemiddelde prefixinvocaties per expert |
|---|---:|
| DeepSeek-V2-Lite | 96 |
| Qwen3-30B-A3B | 64 |
| Kimi-Linear-48B-A3B | 32 |

Qwen levert dus de noodzakelijke hogere-E-test met een factor 1,5 minder
waarnemingen per expert dan V2, gebruikt een native Transformers-architectuur
en vraagt 37,2 GB minder repositoryopslag dan Kimi. Kimi is wetenschappelijk
een sterkere extreme-E-conditie, maar is niet de meest efficiënte eerste
falsificatietest. V4-Flash blijft volgens het bronprotocol pas na deze stap aan
de orde.

De algebraïsche resident-atlasbound bij `T=1024` is voor Qwen
`(2d + 3m) * kT = 52.428.800` elementen, of 100 MiB in BF16 per laag. De
volledige BF16-expertmatrices zijn 1.207.959.552 bytes per laag. Die 11,52×
verhouding is uitsluitend een opslagbound; de P1-cold-bytegate telt optimistic
packed-int4 misses en bewijst geen runtime of resident geheugengebruik.

## Risico- en stopbesluit

- Alleen het BF16 Base-checkpoint wordt gebruikt; geen FP8/GPTQ-afgeleide,
  omdat die een andere numerieke teacher-trajectory kan geven.
- Download vereist minimaal 90 GiB vrije schijfruimte. De machine had vóór
  preregistratie 338.450.546.688 vrije bytes.
- De capture moet shard-/laagstreaming gebruiken. Limieten: proces-RSS
  maximaal 32 GiB en GPU-geheugen maximaal 7,5 GiB.
- Als de geldige hogere-E-test bij rank 32 opnieuw minder dan 80%
  double-fast haalt na representatieve prefill, is hard-falsificatieregel 1
  gereproduceerd op V2 én een hogere-E-model. `RSIV_MOE_V1` sluit dan als
  `falsified_rank_working_set`; Kimi of V4 mag dat resultaat niet achteraf
  wegselecteren.

## Besluitgrens

Qwen mag pas na de afzonderlijke P1C-preregistratie worden gedownload. Een
positieve rankscreen geeft uitsluitend toestemming voor P2; hij is geen
Eureka, snelheidsclaim of kwaliteitsclaim.
