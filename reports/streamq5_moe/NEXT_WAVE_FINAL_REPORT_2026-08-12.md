# STREAMQ5-MoE next-wave — definitief experimenteel rapport

Datum: 2026-08-12  
Bronnen: `intel1.txt`, `intel2.txt`, `NA_ERVF_ZES_HYPOTHESES_2026-08-12.md`

## Uitkomst

De drie documenten bevatten nieuwe bruikbare ideeën, maar geen nieuwe
modelbrede Eureka boven P13. De campagne vond drie sterke, bitexacte
componentoptimalisaties en één positieve exacte 80B-vormpoort. Hun winst
verdunt in de volledige P13-decoder tot minder dan de vooraf vereiste 2% mean.

De oude P13-claim blijft daarom de beste bewezen systeemuitkomst:
Qwen3-30B-A3B, 8-GB RTX PRO 2000 Blackwell Laptop, 4K context, 10.000 tokens,
14,235 tok/s, onder 32 GiB process commit en exact dezelfde voorspellingen,
misses en KV-digest als P12R2.

## Nieuwe positieve resultaten

| Kandidaat | Exactheid | Fysieke uitkomst | Claim |
|---|---|---:|---|
| N1B aligned Q5-loads | 0/1.376.256 bitdiff | Q5-plane p50 `0,8878×` | componentpass |
| N1C reduction-graph autotuner | widths 4/8/16/32/64 exact | Q8 p50 `0,8444×`; Q5 `0,9344×` | componentpass |
| N3A2 concat-QKV | 0/245.760 bitdiff | p50 `0,8817×`, p95 `0,8896×` | componentpass |
| N4B-R 80B synthetic shape | alle widthdigests gelijk | expert p95 `8,869 ms`; conservatief totaal `36,946 ms` | exacte synthetische vormpoort |

N4B-R is onafhankelijk 34/34 geverifieerd. De officiële N4A-vorm bevat
79.674.391.296 parameters; de aligned Q5-bank is 46,497 GiB en past analytisch
met reserve binnen 58 GiB hostgeheugen. Dit rechtvaardigt technisch een echte
Qwen3-Coder-Next-port, maar bewijst geen echte 80B-tokens/s.

## End-to-end-integraties: nuttig maar formeel negatief

| Integratie | mean-ratio | p50-ratio | p95-ratio | Besluit |
|---|---:|---:|---:|---|
| N1BI aligned Q5-load | `0,9864` | `0,9784` | `0,9759` | meanpoort 0,98 gemist |
| N1C2 gemengde Q8/Q5-graaf | `0,9834` | `0,9741` | `0,9613` | meanpoort 0,98 gemist |
| N3A3 concat-QKV | `0,9895` | `0,9840` | `0,9891` | mean/p50-poort gemist |

Alle drie bleven exact voor voorspellingen, misses, KV, LRU en waar van
toepassing volledige logits- en statehashes. De richting is positief, maar het
protocol is niet achteraf versoepeld.

## Belangrijkste falsificaties

- Shared activation: Q5 won geïsoleerd, maar de kandidaat-eerst-replicatie
  keerde de volledige-runtimewinst om.
- Temporal ERVF: same-expert S=4-orakel won sterk, maar echte routerunies waren
  `1,79–2,04×` trager voor S=2/4/8; S=16 liep in resource-spill.
- LM-head: certificerende clusters sloegen 0% rijen over; write-elision schreef
  87,5% minder outputbytes maar was `1,208×` trager.
- MoE down→weight→residual-fusie was `1,412×` trager.
- O-projectie→residual was tijdneutraal/negatief (`0,9994×` p50,
  `1,0047×` p95).
- Exacte sparse temporal Q5 op echte S=4-routes was `1,0457×` trager.

Deze resultaten laten één consistente architectuurregel zien: grote Q5/Q8-
gewichtscans domineren. Fusie of batching helpt alleen wanneer launch/bufferwerk
wordt verwijderd zonder extra synchronisatie, dynamische tokenlussen of
registerspill. Concat-QKV voldoet daaraan; de agressievere fusies niet.

## Context, prefill en geheugen

- BF16 8K past met 33–34 Q5-slots per laag.
- BF16 32K laat 17–18 slots per laag over en breekt static-20; INT8/INT4 lost
  capaciteit analytisch op, maar kwaliteit is niet getest.
- Service-ready TTFT voor het fysieke 7-tokenprompt: `459,244 ms`;
  inclusief domeinactivatie: `655,005 ms`.
- Sequentiële 4K-input kost `284,307 s` (`14,407` effectieve input tok/s).
  Een echte GEMM-prefillruntime blijft een afzonderlijk project.

## Volledigheid en grenzen

De next-wave-registry bevat 33 items en heeft nul `queued` of `in_progress`.
Niet lokaal uitvoerbare varianten zijn expliciet `blocked_scope`,
`blocked_artifact` of `blocked_hardware`, niet als negatief vermomd:

- cp.async/TMA en een nieuwe getegelde Q5-bank;
- reduction-graphcompiler over INT4/BF16/RMSNorm/softmax;
- echte 8K/32K-runtime en GEMM-prefill met verse kwaliteitssets;
- volledige DeepSeek-V2-Lite- en Qwen3-Coder-Next-runtimes;
- identieke GemLite/CUTLASS/QUICK-adapters, tweede GPU en drafter/MTP.

## Eindverdict

De nieuwe documenten leverden aantoonbaar betere lokale kernels en maakten de
80B-port technisch aannemelijk. Ze leverden **geen nieuwe bewezen LLM-
werelddoorbraak**: er is geen sterke equivalente externe GPU-baseline, tweede
GPU, echte 80B-runtime of brede modelreproductie. Het wetenschappelijk sterkste
resultaat blijft de exacte virtualisatie van de originele floating-point-
reductiegraaf; N1C generaliseert die nu binnen Q8/Q5-GEMV-vormen.

De rationele volgende grote stap is niet nog een kleine P13-kerneltweak, maar
een echte Qwen3-Coder-Next-runtimeport of een equivalent-semantische publieke
GPU-baseline. Beide zijn grotere, expliciet afgebakende projecten.
