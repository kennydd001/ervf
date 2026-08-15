# N4B-R — exacte synthetische 80B-replicatie: onafhankelijk rapport

## Verdict

**Pass: 34/34 onafhankelijke controles slagen.** De N4B-methodologische fout is
hersteld, de evaluator en alle bronartefacten zijn cryptografisch gebonden, en
de volledige synthetische outputdigest is CPU-only onafhankelijk gereconstrueerd.

Het formele verdict is:

> `independently_verified_exact_synthetic_shape_timing_pass`

Voor deze audit is geen GPU-run uitgevoerd.

## Herstelde exacte semantiek

De evaluator bevat nu de canonieke STREAMQ5-SwiGLU-volgorde:

```text
silu = round_bf16(g / (1 + exp(-g)))
out  = round_bf16(silu * up)
```

De oude niet-canonieke expressie zonder tussentijdse SiLU-afronding is afwezig.
Daarmee is de in de N4B-audit gevonden semantische fout hersteld.

## Cryptografische reproduceerbaarheid

De volgende bindings zijn tegen de actuele bestanden gecontroleerd en exact:

- preregistratie-SHA256;
- evaluator-SHA256;
- oorspronkelijke N4B-SHA256;
- N4A- en N1C-SHA256;
- synthetische input-SHA256 uit seed `120825`.

De input is onafhankelijk met NumPy opnieuw gegenereerd. De herberekende digest
is exact:

```text
28c9b40609a3e2b76c070c0dba794a5f0f5181f6b25fa97ca3565db560f504f1
```

## Onafhankelijke reconstructie van de volledige outputdigest

De synthetische bank is deterministisch: Q5-payloadbytes zijn `0x55` en alle
BF16-scales `0x3c00`. Daardoor kon de audit CPU-only reconstrueren:

1. de little-endian acht-codes-in-vijf-bytes Q5-decode;
2. BF16-weightafronding;
3. de volledige width-16 virtuele 256-lane reductieboom met FP32-stapafronding;
4. de canonieke tweestaps-BF16-SwiGLU;
5. de downprojectie;
6. alle gate/up/down-outputs voor elf experts × 48 lagen.

De gereconstrueerde uniforme tussenwaarden zijn:

| waarde | float32/BF16-resultaat |
|---|---:|
| raw gate/up | -2,8125 |
| afgeronde SiLU | -0,1591796875 |
| gate na SwiGLU | 0,447265625 |
| down | 0,89453125 |

De onafhankelijk berekende digest van alle **1.622.016 float32-outputwaarden**
is:

```text
80097faf092ca9ba4aadf0ce6609046192c2350578c4c05a7dd6ac2bad56df0b
```

Deze digest is exact gelijk aan `16_reference`, width 8, width 16 en width 32
in het resultaat. Daarmee is de width-exactheid nu daadwerkelijk onafhankelijk
controleerbaar en bevestigd.

## Record- en vormcontract

| onderdeel | onafhankelijk herberekend |
|---|---:|
| gewichten per matrix | 1.048.576 |
| Q5-codebytes | 655.360 |
| BF16-scale bytes | 16.384 |
| header / alignmentpadding | 64 / 4.032 bytes |
| matrixrecord | 675.840 bytes |
| expertrecord | 2.027.520 bytes |
| fysieke slots | 528 |
| residentiële bank | **1.070.530.560 bytes** |
| outputs per laag | 33.792 |
| outputs over 48 lagen | **1.622.016** |

Alle waarden zijn exact gelijk aan het evaluator- en resultaatcontract.

## Raw eventstatistieken en selectie

Alle opgeslagen eventarrays zijn opnieuw geaggregeerd. Mean, p50, p95, minimum
en maximum zijn exact gelijk aan het resultaat; validation bevat 30 en test 120
samples per variant.

### Validatie

| width | p50 ms | p95 ms |
|---:|---:|---:|
| 8 | **6,86634** | 12,97413 |
| 16 | 7,45618 | 20,38078 |
| 32 | 12,10478 | 27,76894 |

Alle widths zijn exact en width 8 heeft de laagste p50. De selectie reproduceert
daarom exact.

### Nieuwe AB/BA-test

| width | mean ms | p50 ms | p95 ms | min–max ms |
|---:|---:|---:|---:|---:|
| 16 | 7,84449 | 7,57824 | 9,33130 | 7,16800–10,27722 |
| 8 | **7,42153** | **7,15070** | **8,86889** | 6,70854–9,85254 |

De testarrays verschillen van N4B en bevatten de vooraf geregistreerde 120
nieuwe metingen per variant. De fysieke residentiële expert-computepoort is
`8,86889 ≤ 50 ms`.

## Dense- en totaalprojectie

De exacte herberekening gebruikt N1C Q8-p95 `9,066176 ms`, bronvolume
`1.248.931.840` bytes en de officiële N4A Q8-shell van `1.933.921.280` bytes:

| projectie | p95 ms |
|---|---:|
| byte-lineaire dense shell | 14,038613 |
| conservatieve 2× dense shell | **28,077227** |
| resident expert + dense 2× | **36,946119** |

Alle negen N4B-R-gates zijn exact herberekend en slagen, inclusief gelijke
outputdigests en de N4A-host/4K/32K-capaciteitsvoorwaarden.

Als gevoeligheidscontrole is ook N4A's analytische all-cold H2D-tijd van
37,202202 ms toegevoegd:

| scenario | expert ms | totaal met dense 2× |
|---|---:|---:|
| ideale DMA/compute-overlap | 37,20220 | 65,27943 |
| volledig serieel | **46,07109** | **74,14832** |

Ook de seriële analytische variant blijft onder de 50/90-ms-poorten. Dit blijft
een projectie en geen fysieke H2D-p95-meting.

## Wat hiermee bewezen is

N4B-R bewijst op de lokale RTX PRO 2000 Blackwell Laptop GPU dat een volledig
residentiële synthetische Q5-expertplane met de officiële Qwen3-Coder-Next-vorm
— 48 lagen, top-10 routed plus één shared expert, hidden 2048 en intermediate
512 — exact over ERVF-widths 8/16/32 uitvoerbaar is. Width 8 haalt fysiek een
p95 van 8,869 ms voor deze residentiële actieve expertcompute. De bijbehorende
offload- en Q8-shellbudgetten blijven analytisch ruim binnen de vooraf
vastgelegde 50/40/90-ms-poorten.

Dit is voldoende om de volgende fysieke port-/checkpointstap rationeel te
autoriseren binnen de preregistratiegrens.

## Wat hiermee niet bewezen is

N4B-R bevat nog steeds:

- synthetische, identieke Q5-recordpayloads in plaats van echte 80B-gewichten;
- geen echt checkpoint of modelkwaliteit;
- geen echte Qwen3-Coder-Next-routersporen of cachemissverdeling;
- geen fysieke H2D-p95 of bewezen DMA/compute-overlap;
- geen fysieke Q8-80B-shellmeting;
- geen Gated DeltaNet-, attention-, expertmix- of volledige decodertijd;
- geen prefill, thermische duurtest of end-to-end tokens/s.

Daarom bewijst N4B-R niet dat het volledige 80B-model al op 8 GB VRAM draait of
≥10 tok/s haalt. Het bewijst de exacte synthetische actieve expertvorm en de
vooraf geregistreerde analytische capaciteits-/rooflinepoort.

## Artefacten

- preregistratie: `N4BR_SYNTHETIC_80B_EXACT_REPLICATION_PREREGISTRATION.md`;
- evaluator: `scripts/streamq5_moe/run_n4br_synthetic_80b_exact_replication.py`;
- resultaat: `n4br_synthetic_80b_exact_replication.json`;
- onafhankelijke verifier:
  `scripts/streamq5_moe/verify_n4br_synthetic_80b_exact_replication.py`;
- verificatie-JSON:
  `n4br_synthetic_80b_exact_replication_verification.json`.
