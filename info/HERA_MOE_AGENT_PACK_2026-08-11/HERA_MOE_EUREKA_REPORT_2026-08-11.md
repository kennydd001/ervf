# HERA-MoE — Hot Entropy-Resident, Rare-Exact

**Datum:** 2026-08-11  
**Bron-SHA-256:** `8e6a2951a6ed4bd11ac3094262e53dbc075f3771c7e33279643e49d2f7bba070`  
**Status:** system-level Eureka hypothesis; quality, actual packing and runtime remain unproven.

## Uitkomst in één zin

De coverage-negative E2GQ-run blijkt niet alleen een blokkade, maar ook een
hardwarepartitionering te bevatten: de 4.449 voldoende gedekte experts kunnen
als entropy-coded 2-bit GPTQ-resident set in VRAM passen, terwijl de 1.695
ondergedekte experts exact in BF16 in systeem-RAM kunnen blijven. Op de locked
WikiText-routing vormen die cold experts 27,59% van de expertbank, maar slechts
0,3811% van de expertinvocaties.

De fundamentele correctie is:

> We hoeven niet de volledige expertbank onder 2 bpp te krijgen. Het doel is de
> veelgebruikte working set in 8 GiB VRAM te krijgen en zeldzame, slecht
> calibreerbare experts exact uit de 32-GiB hosttier te serveren.

## 1. Exacte partitionering

| Grootheid | Waarde |
|---|---:|
| Laag-expertparen | 6,144 |
| Hot, count ≥128 | 4,449 |
| Cold, count <128 | 1,695 |
| Nooit geraakt | 196 |
| Cold parameterfractie | 27.588% |
| Cold selectiefractie op locked corpus | 0.381136% |
| Verwachte cold calls per outputtoken | 1.463562 |
| Hot experts per laag | 83–116 |
| Slechtste laag-coldfractie | 0.645065% |

De selectiefractie is géén router-probability-mass en geen cross-domain
garantie. Zij is alleen een exacte top-8-count uit de locked WikiText-run.

## 2. Geheugenprojectie

Aannames:

- hot experts: diagnostische entropy-GPTQ-rate `1.930709` bpp;
- non-expertweights: INT4;
- cold experts: BF16, dus geen calibratie- of quantisatiefout.

| Tier | Grootte |
|---|---:|
| Hot entropy-GPTQ experts in VRAM | 4.718 GiB |
| Non-expert INT4 in VRAM | 0.718 GiB |
| Totaal resident weight-VRAM | **5.436 GiB** |
| Over binnen 8 GiB vóór KV/buffers | **2.564 GiB** |
| Cold BF16-bank in host-RAM | 14.897 GiB |
| Alle weights over beide tiers | **20.334 GiB** |

Dit past als gewichtenset ruim binnen 32 GiB host-RAM plus 8 GiB VRAM. Een
echte runtime kan door page cache, pinned buffers, coderstate en KV-cache
meer gebruiken; dat moet fysiek worden gemeten.

## 3. Cold-transferprojectie

Eén BF16-expert is 9.000 MiB. Op de locked routing:

```text
48 lagen × 8 calls × 0.003811359406
= 1.463562 cold calls/token
```

Daaruit volgt:

| Cold tier | MiB/token | GiB/s bij 10 tok/s |
|---|---:|---:|
| BF16 exact | 13.172 | 0.1286 |
| W4 | 3.396 | 0.0332 |

Zelfs BF16-coldtraffic is qua gemiddelde bandbreedte klein. De echte risico's
zijn domain shift, burstiness, PCIe-launchlatency en p95/p99-coldcalls.

## 4. Waarom dit sterker is dan EFCQ

EFCQ probeerde de hele bank kunstmatig onder 2 bpp te houden. Daardoor moest
ook de calibratiestaart een lage-precisiefallback krijgen.

HERA gebruikt de werkelijke hardwarecontracten:

```text
VRAM <= 8 GiB
host RAM <= 32 GiB
```

en kan de ondergedekte experts exact houden. Dat:

1. verwijdert het onopgeloste coverage-free quantisatieprobleem;
2. voorkomt dat WikiText-zeldzaamheid wordt vertaald naar lage precisie;
3. maakt kwaliteit afhankelijk van de goed calibreerbare hot tier;
4. gebruikt routerbekendheid om cold weights asynchroon te prefetchen.

## 5. Mogelijke kwaliteitscorrectie zonder de VRAM-doorbraak te verliezen

Wanneer hot 2-bit GPTQ een near miss is, kost een rank-8 INT4-corrector over
alle hot matrices vóór metadata slechts:

```text
0.140 GiB
```

Rank 16 en 32 kosten respectievelijk
`0.280` en
`0.560` GiB.

Dit is geen autorisatie voor een ranksweep. Eén correction rank moet vóór
held-out evaluatie worden vastgelegd.

## 6. Preregistered falsificatiepad

### P0 — onafhankelijke tier-audit

- reproduceer alle counts;
- verzamel naast counts ook `sum(p_e)`, `sum(p_e²)`, margins en cold calls per
  token;
- meet general text, code, math, multilingual en instruction corpora;
- freeze één hotset zonder testdata.

### P1 — full-model quality, geen runtime

Vergelijk:

1. BF16 teacher;
2. hot fixed-width 2-bit GPTQ + cold BF16;
3. hot entropy-exact GPTQ + cold BF16.

De twee GPTQ-paden moeten dezelfde gewichten voorstellen.

Gate:

- relatieve CE ≤2%;
- geen domein met catastrofale regressie;
- 512-token onafhankelijke rollouts stabiel.

Bij CE >2% maar ≤10% mag exact één vooraf geregistreerde rank-8 correction
worden getest. Bij >10% sluit de kwaliteitshypothese.

### P2 — werkelijk hot pack

- actual hot expertbestand ≤4,95 GiB inclusief scales, tables, offsets en
  alignment;
- bit-exact code/scale-decode;
- random access per expert/matrix/chunk.

### P3 — cold-tier microbenchmark

- pinned/mmap BF16 cold bank;
- async expertprefetch direct na routerlogits;
- 0-, 0,5- en 1,0-GiB cold cache als vooraf vastgelegde cells;
- meet per-call latency en p50/p95/p99 coldcalls/token.

### P4 — fused full runtime

Baseline:

- true fixed uint2 hot experts;
- dezelfde cold BF16-tier;
- dezelfde trunk en KV-configuratie.

Finale gates:

```text
VRAM <= 8 GiB
process RAM <= 32 GiB
batch-1 decode >= 10 tok/s
relative CE <= 2%
512-token rollouts stable
```

### P5 — tweede MoE-familie

Geen algemene claim vóór replicatie.

## 7. Claimboundary

Expertcaching, CPU/GPU-offload, mixed precision en entropy decoding zijn prior
art. De verdedigbare nieuwe bevinding is voorlopig uitsluitend de concrete
Qwen3-working-setintersection:

> Een entropy-coded hot tier van 4,718 GiB plus een INT4 trunk van 0,718 GiB
> past theoretisch in 8 GiB VRAM, terwijl de volledige undercovered expertbank
> exact in 14,897 GiB host-RAM past en op de locked routing slechts 0,3811% van
> de expertcalls veroorzaakt.

Het is nog geen kwaliteits-, snelheids- of nieuwheidsbewijs.
