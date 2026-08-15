# ERGV-C1 — generated width-16 GPU-equivalentie

Datum: 2026-08-12  
Verdict: **PASS**

## Kernresultaat

De restricted ERGV-codegenerator heeft zelfstandig Q8- en Q5-width-16
row-reducers geproduceerd. Die gegenereerde code is samen met de
handgeschreven P7-bron in één CUDA-module gecompileerd en op dezelfde
synthetische fysieke inputs uitgevoerd.

| Metriek | Resultaat |
|---|---:|
| CUDA-compileduur | 5,023 s |
| Vergelijkingsgroepen | 20 |
| Vergeleken outputelementen | 115.496 |
| Verschillende bits | **0** |
| Maximale absolute fout | **0** |
| Alle outputs eindig | ja |
| Totale geldige run | 5,101 s |

De gate gebruikte vier vaste inputfamilies: random, nul, alternerende schaal
en cancellation. De Q8-vormen waren `137×2048` en `65×4096`. Q5 draaide acht
fysieke synthetische records voor gate, up en down met respectievelijk de
bestaande 2048- en 768-kolomsemantiek.

## Betekenis

C1 sluit de belangrijkste kloof van de CPU-proef: de gegenereerde width-16
bron is niet alleen logisch isomorf, maar compileert werkelijk met NVRTC en
reproduceert op de lokale Blackwell-GPU de bits van manual P7 voor Q8 én Q5.
De dunne ABI-wrappers bevatten alleen indexering en outputopslag; de loads,
MAC-volgorde en reductie zitten in de gegenereerde helpers.

## Omgevingscontrole

Een eerdere start met WindowsApps-Python stopte vóór CUDA omdat die interpreter
geen CuPy bevatte. De geldige run gebruikte de bestaande projectruntime
`.venv\Scripts\python.exe` met CuPy 14.1.1. De misstart heeft geen output
geopend en staat apart beschreven in
`ERGV_C1_ENVIRONMENT_START_FAILURE_2026-08-12.md`.

## Bewijsgrens

Er is bewust geen snelheid gemeten. Een C1-pass bewijst daarom niet dat
gegenereerde code manual P7 verslaat. Verder zijn echte modelbanken, breedtes
4/8/32/64 op GPU, een tweede checkpoint, een tweede GPU-architectuur en
publieke equivalente kernels nog niet getest. Dit resultaat rechtvaardigt de
volgende compilerfase, maar nog geen industriële of nieuwheidsclaim.

## Artefacten en provenance

- Preregistratie:
  `reports/streamq5_moe/ERGV_C1_GENERATED_GPU_PREREGISTRATION.md`
- Runner:
  `scripts/streamq5_moe/ergv_c1_generated_gpu_gate.py`
- Machineleesbare uitvoer:
  `reports/streamq5_moe/ergv_c1_generated_gpu_gate.json`
- Preregistratie-SHA-256:
  `df689e8b5bebe82397688cc5ebec8a16d50a96c9f928155558b6e288e54e7776`
- Compiler-SHA-256:
  `6b87e672f392cee34b729726c69a75ab58520440bd5ab20e90bac5be015555c2`
- Runner-SHA-256:
  `7dcea424437be9b96d35d8042b10be8f5b5e2a18d3fff1696f9feec45cceebac`
- Gegenereerde CUDA-SHA-256:
  `f9be47ec26ff141c3202c09df8adcf2ca9f52bf2bca87cc1b156e277a6a451ae`
- Gecombineerde CUDA-SHA-256:
  `02f0e209184c11181722fbef7458218bbf12c036679722217a58d2339ebe3e01`

