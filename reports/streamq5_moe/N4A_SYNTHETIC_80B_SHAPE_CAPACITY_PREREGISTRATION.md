# N4A — synthetic Qwen3-Coder-Next 80B shape/capacity preregistratie

Vastgelegd vóór het produceren van het N4A-resultaat. Deze fase gebruikt alleen
CPU, lokale rekenregels en officiële checkpointmetadata; er wordt geen GPU-kernel
gestart en er worden geen gewichtspayloads gedownload.

## Hypothese

De officiële vorm van `Qwen/Qwen3-Coder-Next` kan met de bestaande STREAMQ5-
recordsemantiek binnen de lokale harde budgets worden afgebeeld:

- 48 lagen, 512 routed experts, top-10;
- hidden size 2048, routed en shared intermediate size 512;
- één shared expert per laag;
- Q5 routed/shared expertrecords: 5-bit codes, BF16 group-128-scales,
  64-byte header en 4096-byte recordalignment;
- Q8 dense matrices: INT8-codes plus BF16 group-128-scales; kleine vectoren en
  de depthwise convweight blijven BF16;
- Q8-LM-head resident op de GPU en Q8-embedding persistent in host-RAM.

## Bevroren officiële bron

- Model: `Qwen/Qwen3-Coder-Next`.
- Revisie: `a19358a7659bd1f564300250ee189120c49a562f`.
- De officiële `config.json` en de safetensorsheaders zijn gezaghebbend.
- Het script genereert alle verwachte 74.391 sleutels en vormen uit de config en
  eist exacte gelijkheid met de officiële headers.

De hybride laagvolgorde is driemaal Gated DeltaNet gevolgd door één full-
attentionlaag, dus 36 lineaire en 12 full-attentionlagen. De stateberekening
gebruikt BF16 voor full-attention KV en convstate, en FP32 voor de recurrente
DeltaNet-matrix, conform het referentiepad dat de recurrente toestand in FP32
opbouwt.

## Lokale hardware- en budgetaannames

- Fysieke VRAM: het in P7C gemeten totaal (`8.546.484.224` bytes).
- CUDA/contextreserve: P7C `total_vram_bytes - free_before_bytes`.
- Extra scratch/stagingreserve: 256 MiB.
- Shared Q5-bank resident op de GPU; alleen routed experts gebruiken de
  resterende expertcache.
- Hostlimiet: 58 GiB, inclusief 1 GiB expliciete procesreserve bovenop Q5-bank,
  Q8-embedding en acht pinned stagingvensters.
- Contexten: 4096 en 32768 tokens, batch 1.

## Vooraf vastgelegde poorten

De CPU shape/capacity-gate slaagt alleen wanneer alle volgende voorwaarden waar
zijn:

1. officiële sleutels en vormen zijn exact gelijk aan de gegenereerde set;
2. het checkpoint telt exact 79.674.391.296 parameters;
3. de fysieke routed-plus-shared Q5-bank is hoogstens 50 GiB;
4. de Q8-device-shell inclusief LM-head is hoogstens 2,0 GiB;
5. het hostbudget inclusief 1 GiB reserve blijft hoogstens 58 GiB;
6. bij zowel 4K als 32K blijven na alle reserves minstens 32 routed
   expertrecords per laag cachebaar;
7. de actieve routed-plus-shared expertmassa is kleiner dan de routed
   expertmassa van de bestaande Qwen3-30B-A3B-runtime.

## Niet geopend in deze fase

De fysieke performancepoort blijft gesloten. N4A meet hier dus niet:

- ERVF-width 8/16/32 op intermediate 512;
- expert-p95 ≤50 ms, dense-shell-p95 ≤40 ms of totaal-p95 ≤90 ms;
- routinglokaliteit, kwaliteit, prefill, 32K-attentiontijd of end-to-end tok/s.

Een CPU-pass autoriseert alleen een afzonderlijk vooraf geregistreerde
synthetische GPU-shapebenchmark. Hij autoriseert geen volledige checkpointdownload
en bewijst niet dat 80B op 8 GB al ≥10 tok/s haalt.
