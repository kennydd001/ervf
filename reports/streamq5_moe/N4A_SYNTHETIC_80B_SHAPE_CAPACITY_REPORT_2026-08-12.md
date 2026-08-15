# N4A — Qwen3-Coder-Next 80B CPU shape/capacity-rapport

## Uitkomst

**Pass voor de vooraf geregistreerde CPU shape/capacity-gate; fysieke
performance blijft onbewezen.** Alle 74.391 verwachte tensors en hun vormen zijn
exact teruggevonden in de officiële safetensorsheaders van revisie
`a19358a7659bd1f564300250ee189120c49a562f`. Het checkpoint telt exact
79.674.391.296 BF16-parameters (159.348.782.592 payloadbytes). Er is geen
gewichtspayload gedownload en geen GPU-kernel gestart.

Officiële bronnen: [bevroren config](https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/a19358a7659bd1f564300250ee189120c49a562f/config.json) en het
[Qwen3-Next-referentiepad](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py).

## Exact geverifieerde vorm

| eigenschap | resultaat |
|---|---:|
| decoderlagen | 48 |
| Gated DeltaNet / full attention | 36 / 12 |
| routed experts per laag | 512 |
| actieve routed experts | 10 |
| hidden / expert-intermediate | 2048 / 512 |
| shared experts | 1 per laag, intermediate 512 |
| vocabulary | 151.936 |
| officiële tensors | 74.391 |
| officiële parameters | 79.674.391.296 |

De gegenereerde sleutelset, tensorvormen, routed-tensorcount (`73.728`) en
shared-tensorcount (`144`) zijn exact gelijk aan de officiële headers.

## Q5 expertbank

De STREAMQ5-recordindeling levert voor elk van gate, up en down:

- 1.048.576 gewichten;
- 655.360 bytes 5-bit codes;
- 16.384 bytes BF16 group-128-scales;
- 64 bytes header en 4.032 bytes padding;
- 675.840 bytes per matrixrecord, 2.027.520 bytes per expert.

Daaruit volgt:

| post | bytes | GiB |
|---|---:|---:|
| 48 × 512 routed experts | 49.828.331.520 | 46,387 |
| 48 shared experts | 97.320.960 | 0,091 |
| volledige aligned Q5-bank | **49.925.652.480** | **46,497** |

Per token zijn 1.509.949.440 routed en 150.994.944 shared expertgewichten
actief: samen 1.660.944.384. Dat is 91,67% van de 1.811.939.328 routed actieve
expertgewichten in de bestaande Qwen3-30B-A3B-baseline. Een volledig koude
top-10 route vraagt 973.209.600 aligned H2D-bytes per token, analytisch 37,20 ms
bij de lokaal gemeten 26,16 GB/s.

## Host- en VRAM-budget

Hostaccounting:

| post | GiB |
|---|---:|
| volledige Q5-bank | 46,497 |
| persistente Q8-embedding | 0,294 |
| acht pinned stagingvensters | 0,015 |
| expliciete procesreserve | 1,000 |
| totaal | **47,806** |
| headroom binnen 58 GiB | **10,194** |

De exact uit de checkpointvormen berekende Q8 device-shell is 1.801 GiB:
1.507 GiB dense core plus 0.294 GiB LM-head. Dat is aanzienlijk lager dan de
eerdere ruwe 2,54-GiB-projectie.

Voor de deviceberekening zijn het fysiek door P7C gemeten VRAM-totaal en de
gemeten CUDA/contextreserve gebruikt, plus 256 MiB extra scratch/staging en de
volledige shared Q5-bank resident:

| context | full-attention KV | vaste DeltaNet-state | routed cache | slots | slots/laag |
|---|---:|---:|---:|---:|---:|
| 4K | 96 MiB | 74,25 MiB | **4,570 GiB** | 2.420 | 50,42 |
| 32K | 768 MiB | 74,25 MiB | **3,914 GiB** | 2.072 | 43,17 |

Beide contexten halen de vooraf geregistreerde capaciteitspoort van minstens 32
routed records per laag.

## Betekenis en volgende beslissende test

Dit resultaat sluit de eenvoudige geheugen- en vormbezwaren tegen een 80B-port
uit. Het ondersteunt de Active-Set-Invariance-hypothese structureel: 2,62× meer
totale modelcapaciteit gaat hier samen met 8,33% minder actieve expertgewichten
dan de bestaande 30B-routed baseline wanneer de shared expert wordt meegeteld.

Het bewijst nog niet dat de hybride dense shell snel genoeg is of dat ERVF-width
16 optimaal blijft bij intermediate 512. De eerstvolgende toegestane stap is
daarom een afzonderlijk vooraf geregistreerde synthetische GPU-shapebenchmark
voor widths 8/16/32, shared expert, full-attention/Gated-DeltaNet-projecties en
zero-/plausible-cacheverkeer. De harde fysieke poorten blijven:

- expert-p95 ≤50 ms;
- dense-shell-p95 ≤40 ms;
- geprojecteerd totaal-p95 ≤90 ms.

Pas een pass daarop rechtvaardigt de volledige checkpointdownload.

## Artefacten en claimgrens

- preregistratie: `N4A_SYNTHETIC_80B_SHAPE_CAPACITY_PREREGISTRATION.md`;
- evaluator: `scripts/streamq5_moe/run_n4a_synthetic_80b_shape_capacity.py`;
- machineleesbaar resultaat: `n4a_synthetic_80b_shape_capacity.json`.

Claimgrens: CPU-only, officiële metadata en analytische geheugenaccounting. Geen
modelkwaliteit, routerspoor, prefill, 32K-attentiontijd, kernelmeting,
end-to-enddecode of ≥10 tok/s-claim.
