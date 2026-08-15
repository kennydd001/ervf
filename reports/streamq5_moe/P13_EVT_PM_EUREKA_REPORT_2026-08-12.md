# P13 EVT-PM + Pinned-Window Streaming — lokaal Eureka-rapport

Datum: 2026-08-12

## Resultaat

De combinatie van drie bitexacte systeemtransformaties sluit de eerder open
32-GiB/4K/duurpoort:

1. **Pinned-Window Streaming:** 17,367 GiB experts blijven read-only mapped;
   acht pinned vensters (24.281.088 bytes) verzorgen asynchrone misskopieën.
2. **Q8 Residency-Lifetime Staging:** alleen de 316.026.880-byte embedding blijft
   host-resident; Q8-trunkrecords worden via maximaal 8.519.680 bytes staging
   naar hun definitieve GPU-residentie gebracht.
3. **Explicit-Add Virtual-Tile + Probability Materialization (EVT-PM):** acht
   warps verwerken acht attentionposities per block terwijl iedere warp de vier
   oorspronkelijke virtuele threadgroepen en de exacte FP32-boom emuleert.
   Softmaxprobabilities worden één keer BF16-afgerond en daarna door alle 128
   value-dimensies hergebruikt.

## Geïsoleerd attentionbewijs

Nul bitverschillen op scores én attentionoutputs voor alle 48 lagen bij context
128, 512, 1024 en 4096. Ongeopende test:

| Context | Origineel p50 | EVT-PM p50 | Ratio | p95-ratio |
|---:|---:|---:|---:|---:|
| 1024 | 24,274 ms | 3,630 ms | 0,1495 | 0,1533 |
| 4096 | 96,626 ms | 12,929 ms | 0,1338 | 0,1429 |

## Volledige 10K-tokenrun onder 32 GiB

| Metriek | P12R2 zonder EVT-PM | P13C EVT-PM |
|---|---:|---:|
| tokens | 10.000 | 10.000 |
| throughput | 9,043 tok/s | **14,235 tok/s** |
| mean | 110,201 ms | **69,862 ms** |
| p95 | 166,547 ms | **91,984 ms** |
| p99 | 180,870 ms | **100,498 ms** |
| peak process commit | 10,180 GB | **10,185 GB** |
| 4K KV | 402.653.184 bytes | **exact gelijk** |
| contextgepaarde thermiek | niet vooraf bruikbaar | **1,083 mean / 1,061 p95** |

Prediction-digest, alle 10.000 missenaantallen en de volledige 4K-KV-digest
zijn exact gelijk aan P12R2. Alle P13C-gates slagen.

## Wat dit bewijst

Op deze RTX PRO 2000 Blackwell 8GB + Core Ultra 9 285H-machine kan de lokale
Qwen3-30B-A3B-base-runtime met fysieke Q5-experts en Q8-trunk onder een harde
32-GiB-processlimiet 10.000 exacte feedbacktokens en volledige 4K-context
afwerken met 14,23 tok/s en p99 onder 110 ms.

## Wat dit niet bewijst

Geen wereldrecord, geen universele nieuwheid, geen winst tegen alle actuele
runtimes, geen generalisatie naar andere modellen/hardware/batches en geen
kwaliteitsclaim buiten de eerder vergrendelde 1.270+1.270-labeltest. Een
same-hardware externe baseline en tweede-modelreplicatie blijven afzonderlijke
voorwaarden voor een brede LLM-doorbraakclaim.
