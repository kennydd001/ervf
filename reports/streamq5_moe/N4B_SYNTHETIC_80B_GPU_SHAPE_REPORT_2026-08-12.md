# N4B — synthetische 80B GPU-vormpoort: onafhankelijk auditrapport

## Verdict

**De numerieke shape/timing-poorten zijn correct herberekend, maar N4B is niet
onafhankelijk als exacte portpoort geverifieerd.** Negentien van 22 auditchecks
slagen. Drie controles falen:

1. de N4B-SwiGLU wijkt af van de bewezen STREAMQ5-BF16-semantiek;
2. het resultaat bevat geen SHA256 van de evaluator;
3. ruwe widthoutputs of per-width-outputdigests zijn niet gearchiveerd.

De juiste status is daarom:

> **gekwalificeerde synthetische shape/timing-pass; exacte portautorisatie
> vereist één gerepareerde, opnieuw vergrendelde run.**

Er is voor deze audit geen GPU-run uitgevoerd.

## Exact herberekende record- en outputcontracten

Voor iedere matrix van `512 × 2048` of `2048 × 512`:

| onderdeel | herberekend |
|---|---:|
| gewichten | 1.048.576 |
| Q5-codes | 655.360 bytes |
| BF16 group-128-scales | 16.384 bytes |
| header | 64 bytes |
| padding tot 4096-byte alignment | 4.032 bytes |
| matrixrecord | **675.840 bytes** |

Drie matrixrecords geven exact **2.027.520 bytes per expert**. Met 48 lagen en
elf adressen per laag zijn er 528 fysieke slots en bedraagt de residentiële
synthetische bank exact **1.070.530.560 bytes**.

Per laag worden `11 × (512 gate + 512 up + 2048 down) = 33.792` waarden
vergeleken. Over 48 lagen is dat exact **1.622.016** waarden, gelijk aan alle
drie comparison-samenvattingen.

## Ruwe eventstatistieken

Alle gemiddelden, percentielen, minima en maxima zijn rechtstreeks uit de
gearchiveerde eventarrays herberekend en zijn bit-voor-bit gelijk aan het JSON.

### Validatie — 30 samples per width

| width | mean ms | p50 ms | p95 ms | min–max ms |
|---:|---:|---:|---:|---:|
| 8 | 6,52410 | **6,32019** | 8,16120 | 6,05040–8,83971 |
| 16 | 6,96396 | 6,78736 | 8,35657 | 6,40013–9,38339 |
| 32 | 11,11440 | 10,49630 | 15,01816 | 9,97795–16,82707 |

Alle widths zijn in het resultaat als bitwise gelijk aan width 16 geregistreerd.
Width 8 heeft onder de vooraf geregistreerde selectieregel de laagste p50 en is
correct geselecteerd.

### Test — 120 samples per variant

| width | mean ms | p50 ms | p95 ms | min–max ms |
|---:|---:|---:|---:|---:|
| 16 | 7,29094 | 7,08584 | 8,28345 | 6,78947–8,38586 |
| 8 | **6,85787** | **6,69968** | **7,70669** | 6,42547–8,16243 |

De gerapporteerde fysieke resident-computepoort `7,70669 ≤ 50 ms` is dus
rekenkundig correct.

## Dense-shellprojectie en poorten

De projectie gebruikt:

- N1C Q8-test-p95: 9,066176 ms;
- fysieke Qwen30-bronbytes: 1.248.931.840;
- officiële N4A 80B Q8-device-shell: 1.933.921.280 bytes.

Daaruit volgt exact:

| projectie | ms |
|---|---:|
| byte-lineaire dense p95 | 14,038613 |
| conservatieve 2× dense p95 | **28,077227** |
| gerapporteerd totaal: resident Q5 + dense 2× | **35,783918** |

Alle acht in N4B gerapporteerde gates zijn correct uit de opgeslagen cijfers en
N4A afgeleid en staan op `true`.

Omdat de 7,70669 ms alleen residentiële Q5-compute meet, is aanvullend de N4A
all-cold H2D-projectie van 37,20220 ms doorgerekend:

| sensitiviteit | expert ms | totaal met dense 2× |
|---|---:|---:|
| ideale DMA/compute-overlap | 37,20220 | 65,27943 |
| conservatief volledig serieel | **44,90889** | **72,98612** |

Ook deze analytische varianten blijven onder respectievelijk 50 en 90 ms. Dit
is geen fysieke p95-meting: H2D-variatie, echte cachemissers en overlap zijn in
N4B niet gemeten.

## Methodologische fout: SwiGLU is niet STREAMQ5-bitexact

De bewezen P6/P7-runtime voert uit:

```text
silu = round_bf16(g / (1 + exp(-g)))
out  = round_bf16(silu * up)
```

N4B voert uit:

```text
out = round_bf16((g / (1 + exp(-g))) * up)
```

De verplichte tussentijdse BF16-afronding ontbreekt. Dat is geen louter
tekstueel verschil. Voor de reeds BF16-representeerbare waarden `g=-7,9375` en
`up=1,5` levert de runtime `-0,0042724609375`, terwijl N4B
`-0,004241943359375` levert.

Alle widths gebruiken dezelfde afwijkende SwiGLU. Daarom bewijst hun onderlinge
gelijkheid niet de in de preregistratie genoemde exacte STREAMQ5-SwiGLU of
bitexactheid tegenover de bestaande runtime. De Q5-matvecreductiegrafen kunnen
onderling wel gelijk zijn; het opgeslagen JSON bevat echter alleen de
comparison-samenvattingen, niet de raw outputs of hun digests. Zonder een nieuwe
GPU-run kan die gelijkheid niet onafhankelijk worden gereconstrueerd.

## Overige auditgrenzen

- Het resultaat bindt preregistratie, N4A en N1C cryptografisch, maar niet het
  evaluatorbestand zelf en evenmin de gegenereerde inputvector.
- De 1,071-GB-bank heeft afzonderlijke adressen maar identieke synthetische
  recordpayloads. Dit test de volledige vorm en residentiële scan, niet echte
  gewichtsdistributies.
- De dense shell is uitsluitend byte-lineair geprojecteerd. Gated DeltaNet-
  recurrence, full-attentioncompute, routing, shared-expertgate, expertmix en
  residuals zijn geen fysieke N4B-metingen.
- Geen echte routersporen, checkpointpayload, kwaliteit, prefill, thermiek of
  end-to-end tokens/s zijn bewezen.

## Vereiste reparatie

Een minimale beslissende N4B-R-run moet vóór uitvoering opnieuw worden
geregistreerd en:

1. de tussentijdse BF16-SiLU-afronding herstellen;
2. de evaluator- en input-SHA256 in het resultaat opnemen;
3. per width een volledige outputdigest archiveren;
4. width 8/16/32 tegen de canonieke width-16-runtime-output vergelijken;
5. dezelfde validation-selectie en onafhankelijke AB/BA-test herhalen.

De huidige timings zijn sterke aanwijzing dat de vorm ruim binnen de analytische
50/40/90-ms-budgetten valt, maar ze mogen niet als afgeronde exacte portpoort of
80B-throughputbewijs worden geregistreerd.

## Artefacten

- bronpreregistratie: `N4B_SYNTHETIC_80B_GPU_SHAPE_PREREGISTRATION.md`;
- bronevaluator: `scripts/streamq5_moe/run_n4b_synthetic_80b_gpu_shape.py`;
- bronresultaat: `n4b_synthetic_80b_gpu_shape.json`;
- onafhankelijke verifier:
  `scripts/streamq5_moe/verify_n4b_synthetic_80b_gpu_shape.py`;
- verificatieresultaat: `n4b_synthetic_80b_gpu_shape_verification.json`.
