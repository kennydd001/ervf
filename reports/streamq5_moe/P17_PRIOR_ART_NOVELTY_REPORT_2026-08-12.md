# P17 prior-art- en nieuwheidsaudit

Datum: 2026-08-12

## Harde conclusie

P13C is een sterke lokale systems-Eureka, maar de huidige bewijsbasis laat
geen claim toe dat dit de snelste, eerste of algemeen beste low-VRAM
MoE-runtime ter wereld is. Expertstreaming, GPU-expertcaches, asynchrone
missafhandeling en IO-bewuste exacte attention hebben duidelijke prior art.

Wat na deze zoekronde mogelijk onderscheidend blijft, is veel smaller: de
combinatie van een gemapte Q5-expertbank met kleine pinned transferwindows én
een attentionkernel die de oorspronkelijke FP32-reductieboom virtueel
emuleert en via expliciete round-to-nearest-adds bitpatroonidentiteit bewaart.
Ik vond geen publicatie met exact die combinatie of met de namen ERVF/EVT-PM.
Dat is negatief zoekbewijs, geen formeel nieuwheidsbewijs.

## Dichtstbijzijnde systemen

1. De publieke `llama.cpp`-discussie over on-demand MoE-paging beschrijft een
   compacte expertslotpool, GPU-gepubliceerde expert-ID's, een CPU-sidecar en
   een gedeeld event. De auteur rapporteert Qwen3-30B-A3B-Q6_K op een M1 Pro
   met 16 GB bij 13 tok/s. Dit maakt de brede claim “eerste expertstreaming op
   kleine hardware” onhoudbaar.
   <https://github.com/ggml-org/llama.cpp/discussions/23324>
2. CPU–GPU Collaborative Inference gebruikt een GPU-expertcache en handelt
   misses asynchroon met CPU-compute af. Dat is zeer nabij de cache/overlap-
   familie, al verschilt de concrete uitvoering.
   <https://arxiv.org/abs/2512.16473>
3. HybriMoE combineert dynamische CPU/GPU-planning, inter-layer prefetching en
   scoregebaseerde caching.
   <https://arxiv.org/abs/2504.05897>
4. OD-MoE laadt experts volledig on demand en gebruikt vooruitvoorspelling op
   gedistribueerde edge-nodes.
   <https://arxiv.org/abs/2512.03927>

## Dichtstbijzijnde attention- en exactheidsprior art

- FlashAttention is sinds 2022 een IO-bewuste, wiskundig exacte getegelde
  attentionmethode. “Exact” betekent daar geen approximatieve attention; het
  document claimt niet bitidentiteit met een specifieke naïeve FP32-boom.
  <https://arxiv.org/abs/2205.14135>
- DASH optimaliseert deterministische FlashAttention-backwardplanning. Het
  bevestigt dat deterministische reducties een actieve onderzoekslijn zijn,
  maar test een andere fase en andere hardware.
  <https://arxiv.org/abs/2601.21824>
- NVIDIA documenteert dat een expliciete `.rn`-roundingmodifier conservatief
  door de optimizer wordt behandeld, terwijl een niet-gemodificeerde add
  agressiever kan worden herschreven. Dat ondersteunt het mechanisme achter
  de P13B-reparatie; de instructiesemantiek zelf is vanzelfsprekend prior art.
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>

## Claims die nu wel en niet verdedigbaar zijn

Wel:

- op de gemeten RTX PRO 2000 Blackwell 8GB-machine voltooide P13C 10.000
  feedbacktokens onder een harde 32-GiB-processlimiet bij 14,235 tok/s;
- de volledige 4K-KV-, prediction- en missdigests bleven exact gelijk aan de
  tragere referentieruntime;
- de geïsoleerde EVT-PM-kernel was op 4K-context bitidentiek over alle 48
  lagen en verlaagde de attentionmedian naar 13,38% van de originele tijd.

Niet:

- wereldrecord of snelste 8GB-runtime;
- eerste expertstreamingsysteem;
- eerste exacte/getegelde attention;
- generalisatie naar andere modellen, GPU's, contexten of batches;
- octrooieerbaarheid of formele afwezigheid van eerdere openbaarmaking.

## Wat voor een bredere claim nog nodig is

Een reproduceerbare externe baseline op dezelfde machine, tweede-modelbewijs,
grotere kwaliteitstest, onafhankelijke code-review en benchmark tegen actuele
low-VRAM-runtimes. Voor een formele nieuwheidsclaim is daarnaast een gerichte
patent- en codebase-search door een deskundige nodig.

