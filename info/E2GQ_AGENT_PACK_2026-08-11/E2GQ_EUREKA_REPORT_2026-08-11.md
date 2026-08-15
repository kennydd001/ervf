# E2GQ-MoE — entropy-exact GPTQ Eureka report

**Datum:** 2026-08-11  
**Bron:** `fleq_moe.zip`  
**Bron-SHA-256:** `a33000d365afa6a7001a645bb112f358f20d4a3d3e6040eeeda9209c8158f6d2`  
**Status:** harde positieve representatiebevinding; end-to-end kwaliteit en runtime nog niet bewezen.

## Uitkomst in één zin

De GSQ-hypothese blijft gefalsificeerd, maar de opslagconclusie voor de
bestaande 2-bit GPTQ-baseline was te pessimistisch: de werkelijke vier
GPTQ-codes zijn sterk niet-uniform en vereisen op de zestien locked experts
ideaal slechts **1.782864891 bpp**. Met de bestaande BF16 group-128 scales is dat
**1.907864891 bpp**, dus reeds onder de harde grens van 2,0 bpp,
zonder één gewicht of modeloutput te wijzigen.

## 1. Gemeten codeverdeling

| Code | Aantal | Fractie |
|---:|---:|---:|
| -2 | 4,713,974 | 6.243883% |
| -1 | 17,846,753 | 23.638875% |
| 0 | 31,599,966 | 41.855661% |
| +1 | 21,336,779 | 28.261581% |

Totaal: **75,497,472 codes**, afkomstig uit 16 experts × 3 matrices.

De nulde-orde Shannon-entropie is:

```text
H(Q) = -Σ p(q) log2 p(q)
     = 1.782864891 bpp
```

De raw BF16 scale-overhead bij group size 128 is exact:

```text
16 / 128 = 0.125000 bpp
```

Daarom:

```text
H(Q) + scales = 1.907864891 bpp
```

De eerdere `2,125 bpp` is correct voor een **fixed-width** 2-bitpack met
raw BF16-scales, maar niet als ondergrens voor een lossless variabele
representatie.

## 2. Geen aggregate toeval

- Alle 16 experts liggen individueel onder 2 bpp.
- Expertbereik: **1.864264–1.921959 bpp**.
- Alle 48 matrices liggen individueel onder 2 bpp.
- Matrixmaximum: **1.928556 bpp**.

Een finite enumerative bound, inclusief 16 bytes header per matrix en raw
BF16-scales, geeft gemiddeld **1.907105 bpp**.

Een bewust conservatieve projectie met 0,01 bpp coderredundantie,
32 bytes header per matrix en 4-KiB-alignment geeft gemiddeld
**1.927951 bpp**, met een expertmaximum van
**1.944444 bpp**.
Dit is nog geen werkelijk geëncodeerd bestand en moet door P1 worden bewezen.

## 3. Exacte progressieve decompositie

Voor iedere GPTQ-code `q ∈ {-2,-1,0,+1}`:

```text
t = max(q, -1)       # ternary core (-1, 0, 1)
e = 1[q == -2]       # zeldzame extreme correction
q = t - e            # exact
```

Gemeten rate:

| Stream | Rate |
|---|---:|
| Ternary core | 1.561892876 bpp |
| Raw BF16 group-128 scales | 0.125000000 bpp |
| Extreme-tail conditional stream | 0.220972015 bpp |
| Exact totaal | 1.907864891 bpp |

Slechts **6.244%** van alle codes heeft de extreme correction.
Dit maakt een exacte resident-core/streamed-tail-runtime mogelijk.

## 4. Qwen3-30B-A3B-projectie

Gebaseerd op de pinned registry:

| Grootheid | Projectie |
|---|---:|
| Routed experts, fixed 2,125 bpp | 7.172 GiB |
| Routed experts, ideale entropy pack | 6.439 GiB |
| Routed experts, conservatieve packprojectie | 6.507 GiB |
| Resident ternary core + scales | 5.693 GiB |
| Volledige extreme tail | 0.746 GiB |
| Ideale routed bank + non-experts BF16 | 9.310 GiB |
| Ideale routed bank + non-experts int4 | 7.157 GiB |
| Conservatieve routed pack + non-experts int4 | 7.224 GiB |

De volledige 8-GB-deployment blijft krap door KV-cache, buffers, coderstate
en niet-expertmetadata. Maar de opslagmuur is niet langer principieel.

Bij 48 lagen, top-8 en drie 768×2048 expertmatrices:

| Routed traffic | MiB/token | GiB/s bij 10 tok/s |
|---|---:|---:|
| Volledige ideale entropy-GPTQ | 412.099 | 4.024 |
| Conservatieve packprojectie | 416.438 | 4.067 |
| Alleen extreme tail | 47.730 | 0.466 |

Deze trafficgetallen zijn fysisch plausibel. Ze bewijzen geen kernel- of
decode-throughput.

## 5. Entropy reserve

De exacte theoretische reserve onder de 2-bpp-gate is:

```text
2.0 - 1.907864891 = 0.092135109 bpp
```

Een rank-8 INT4-corrector per 768×2048 matrix kost vóór metadata:

```text
8 × (768 + 2048) × 4 / (768 × 2048)
= 0.057291667 bpp
```

Dus:

```text
entropy-GPTQ + rank-8 INT4
= 1.965156558 bpp
```

Dat past theoretisch nog onder 2 bpp. Kwaliteitsherstel, factorquantisatie,
scales en werkelijke headers zijn niet bewezen. Dit is één vooraf te
registreren repairroute, niet een uitnodiging tot onbeperkte tuning.

## 6. Nieuwe onderzoeksthese

> **E2GQ-MoE:** behoud de bestaande GPTQ-code assignments exact, codeer ze
> lossless tegen hun werkelijke entropie, en besteed uitsluitend de aantoonbare
> rate-reserve aan één modelbreed getrainde foutcorrector. Decodeer de exacte
> ternary core en extreme tail rechtstreeks in een fused MoE-MVM-kernel.

Dit heropent niet de gefaalde GSQ-lijn. Het is een nieuwe registry met een
andere mechaniek en een andere positieve preconditie.

## 7. Harde fasering

1. **P0 — full-bank entropy census**  
   Alle 48×128 experts, alle matrices. Geen extrapolatie uit 16 experts.

2. **P1 — werkelijk bit-exact bestand**  
   rANS/dtANS, enumerative of een zero/sign/extreme-format. Alle codes en
   BF16-scales moeten bit-exact reconstrueren. Werkelijke bpp inclusief offsets,
   tabellen en alignment ≤1,98.

3. **P2 — drie-laagse GPTQ-oracle**  
   Lagen 0/24/47, exact entropy-pack versus exact fixed GPTQ. Logits moeten
   numeriek dezelfde quantized modelsemantiek hebben.

4. **P3 — full-model GPTQ**  
   Full-depth CE, KL, benchmarks en minstens 512-tokenrollouts.

5. **P4 — alleen bij near miss**  
   Wanneer relatieve CE-schade >2% maar ≤10%: exact één vooraf vastgelegde
   rank-8 INT4 model-wise discrepancycorrector, met werkelijke totale rate ≤2,0.
   Bij >10% stopt de kwaliteitslijn.

6. **P5 — fused decoder/runtime**  
   Vergelijk tegen een true-uint2 fixed pack. Meet werkelijk VRAM, bytes,
   p50/p95-latency, energie en tokens/s. Gate: ≥10 tok/s batch 1.

7. **P6 — tweede familie**  
   Geen algemene claim zonder replicatie.

## 8. Claimboundary

Entropy coding, low-entropy MoE-gewichten, low-rank quantization correction
en fused decoders zijn ieder prior art. De huidige Eureka is daarom:

> **De concrete, vooraf als te groot afgewezen Qwen3 GPTQ-baseline blijkt in
> de eigen locked artifacts lossless onder 2 bpp te passen.**

Niet bewezen:

- full-model GPTQ-kwaliteit;
- werkelijk bestand ≤2 bpp;
- GPU-decodesnelheid;
- 10 tokens/s;
- fundamentele wetenschappelijke nieuwheid.
