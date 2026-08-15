# CORETAIL-MoE — universal resident core, sparse exact tail

**Datum:** 2026-08-11  
**Status:** nieuwe mechanistisch onafhankelijke hypothese; geen bewezen Eureka.

## Uitkomst in één zin

HERA heeft niet aangetoond dat een resident expertbank onmogelijk is. HERA
heeft aangetoond dat **expert-ID** de verkeerde tiering-as is: de vijf
domeinen maken 6.081 van 6.144 laag-expertparen hot. De sterkste nieuwe
architectuur houdt daarom voor **ieder** expert een universele exacte
ternary core resident en streamt uitsluitend een kleine exacte
quantisatieresidual voor de werkelijk gerouteerde experts.

## 1. Wat HERA werkelijk blootlegt

De statische multidomainunion bevat:

```text
6.081 hot / 6.144 totaal = 98,974%
```

De HERA-hotbank kost 6,449 GiB. Lineair doorgetrokken naar alle experts:

```text
6,449 × 6.144 / 6.081 = 6.515813 GiB
```

Alle 63 resterende experts toevoegen kost dus slechts:

```text
68.42 MiB
```

Daarom lost een complexere hot/cold-ID-cache het fundamentele
cross-domainprobleem niet op. De all-expertbank is vrijwel even groot als
de gefaalde union.

## 2. Exacte core/tail-identiteit

De bestaande GPTQ-code is:

```text
q ∈ {-2, -1, 0, +1}
```

Definieer exact:

```text
t = max(q, -1)      # (-1, 0, 1)
e = 1[q == -2]      # sparse extreme correction
q = t - e
```

Voor iedere group-scale `s`:

```text
s*q*x = s*t*x - s*e*x
```

Dit verandert geen GPTQ-code en introduceert geen extra modelfout.

- **Resident core:** ternary `t` plus de originele BF16 group-scales.
- **Exact tail:** alleen de `q=-2`-correcties.
- **Routing:** onveranderd.
- **Domein:** irrelevant; ieder expert heeft zijn core resident.

## 3. Geheugenprojectie

Gebaseerd op de locked E2GQ-codehistogrammen:

| Component | Projectie |
|---|---:|
| Volledige entropy-GPTQ expertbank | 6.439 GiB |
| Ideale ternary core + BF16-scales | 5.693 GiB |
| Eenvoudige zero/sign-coreprojectie | 5.759 GiB |
| Volledige exact extreme tail | 0.746 GiB |
| INT4 non-experttrunk | 0.718 GiB |
| INT8 non-experttrunk | 1.435 GiB |
| BF16 KV-cache, 4K | 0.375 GiB |

### Variant A — alle cores resident

```text
all expert cores       5.759 GiB
INT4 trunk              0.718 GiB
BF16 KV 4K              0.375 GiB
-----------------------------------------
subtotal                6.852 GiB
reported GPU capacity   7.960 GiB
headroom                1.108 GiB
```

De volledige tail blijft in host-RAM. Alleen de actieve tail wordt
overgebracht:

```text
47.73 MiB/token
0.466 GiB/s bij 10 tok/s
```

### Variant B — gate/up-core resident, down-core gestreamd

Om meer ruimte voor kwaliteit en runtimebuffers te creëren:

```text
2/3 core (gate+up)      3.839 GiB
volledige tail          0.746 GiB
INT8 trunk              1.435 GiB
BF16 KV 4K              0.375 GiB
-----------------------------------------
subtotal                6.396 GiB
headroom                1.564 GiB
```

De down-core voor acht actieve experts kan worden gekopieerd terwijl de
resident gate/up-core wordt uitgevoerd:

```text
121.46 MiB/token
1.186 GiB/s bij 10 tok/s
```

Dit is een hardwareprojectie, geen throughputbewijs.

## 4. Waarom deze hypothese HERA overleeft

| HERA static tier | CORETAIL |
|---|---|
| Experts worden op ID hot of cold | Ieder expert heeft een resident core |
| Cross-domainunion groeit tot 98,97% | Geen multidomainunion nodig |
| Cold miss vereist een volledige expert | Alleen een kleine exact tail of down-core wordt geladen |
| Placement hangt af van corpus/domein | De code-decompositie is modelintrinsiek |
| 7,167 GiB vóór KV/workspace | 6,852 GiB inclusief BF16 4K KV in variant A-projectie |

## 5. Dichtste prior art en claimgrens

De brede bouwstenen zijn bezet:

- QMoE: low-entropy MoE-compressie plus bespoke decoders.
- SliceMoE: bit-sliced expert caching en on-demand precision.
- SpQR/SqueezeLLM: dense base plus sparse high-precision residuals.
- MoEpic: expert splitting en overlappende transfers.
- Fast entropy-decoded sparse MVM via dtANS.

CORETAIL mag daarom niet worden verkocht als “de uitvinding van residual
quantization” of “de eerste bit-sliced MoE-cache”.

De te testen systeemclaim is smaller:

> Op een 8-GB single-stream laptop is een **all-expert universal core**
> mogelijk gunstiger dan iedere expert-ID-cache, omdat HERA aantoont dat
> expertidentiteit cross-domain vrijwel geen cold tier overlaat.

## 6. Harde preregistered fasering

### P0 — volledige 6.144-expert census en werkelijk format

Bouw een row-random-access bestand voor:

1. nonzero bitmap;
2. signstream voor nonzeros;
3. originele BF16-scales;
4. entropy-coded extreme flags onder negatieve codes;
5. alle row/block offsets en alignment.

Gates:

- werkelijke core ≤5,95 GiB;
- werkelijke tail ≤0,90 GiB;
- 100% bit-exacte reconstructie;
- variant-A-projectie plus 0,75 GiB verplichte runtime-reserve ≤7,96 GiB;
- geen Shannon-only accounting als eindbewijs.

### P1 — fused matrixkernel

Verplichte baselines:

- BF16;
- fixed uint2 GPTQ;
- entropy-packed exact GPTQ;
- CORETAIL exact.

Meet afzonderlijk gate, up en down op lagen 0/24/47.

Gates:

- exact dezelfde quantized weights;
- geen extra outputfout buiten vastgelegde accumulatietolerantie;
- routed throughput ≥27 miljard weights/s als 1,5× veiligheidsmarge boven
  de 10-tok/s-behoefte;
- tailtransfer en -decode p95 voldoende om binnen 100 ms/token te blijven.

### P2 — quantisatiekwaliteit isoleren

Volledige modellen:

1. BF16 teacher;
2. GPTQ experts + BF16 trunk;
3. BF16 experts + INT4 trunk;
4. GPTQ experts + INT4 trunk;
5. GPTQ experts + INT8 trunk.

Geen runtimeselectie op test.

### P3 — één repair indien near miss

Alleen wanneer relatieve CE-schade >2% maar ≤10%:

- exact één vooraf vastgelegde rank-8 INT4 model-wise correction;
- alle bytes tellen;
- geen rank- of objective-sweep na testinzage.

Bij >10% sluit de kwaliteitslijn.

### P4 — echte memory residency

- werkelijke CUDA-context en allocatorfragmentatie;
- 1K/4K/8K context;
- p50/p95/p99 VRAM;
- variant A versus B;
- process-RAM ≤32 GiB.

### P5 — volledige decode

Eindgates:

- ≤8,0 GiB piek-VRAM;
- ≤32 GiB RAM;
- relatieve CE-schade ≤2%;
- stabiele rollouts van ≥512 tokens;
- ≥10 tokens/s, batch 1;
- tweede MoE-familie vóór een brede claim.

## 7. Harde beoordeling

Dit is nog geen bewezen Eureka. Het is wel de eerste reactie op HERA die
het negatieve resultaat niet probeert weg te tunen:

```text
HERA: identity-tiering is domain-unstable
CORETAIL: do not tier by identity
```

De beslissende proef is nu niet nog een routecount. Het is een werkelijk
all-expert, row-random-access core/tailbestand plus een fused exact kernel.
