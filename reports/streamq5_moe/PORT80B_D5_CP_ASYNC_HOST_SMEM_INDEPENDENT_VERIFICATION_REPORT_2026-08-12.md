# PORT80B-D5 — onafhankelijke CPU-only verificatie

**Verdict:** `verified_strong_transport_component_pass`  
**GPU-context geopend:** nee  
**Alle replaybare checks:** PASS

## Onafhankelijk herberekend

| blocks | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16 | 43.188842 | 43.111248 | 43.926656 | 44.062669 | 42.747295 | 44.096672 |
| 512 | 16 | 43.296984 | 43.156992 | 43.882577 | 44.055184 | 42.859390 | 44.098335 |
| 1024 | 16 | 43.149928 | 43.064224 | 43.476040 | 43.572903 | 42.858559 | 43.597118 |
| 2048 | 16 | 43.214612 | 43.171345 | 43.776559 | 43.997551 | 42.941250 | 44.052799 |

De selection rule kiest correct **1024 blocks**, de laagste validation-p50 (43.064224 ms). Alle vier schedules hebben exact 16 eindige samples. De correctness-token 89.999, validationtokens 90.000–90.015 en testtokens 91.000–91.119 zijn exact en onderling disjunct.

Alle 16 rotatie/omkeerorders matchen en de vier resulterende orderpatronen komen elk viermaal voor. De vastgelegde vier-armalgoritme is echter niet positiegebalanceerd: 256/1024 staan elk achtmaal op posities 0 en 2, terwijl 512/2048 elk achtmaal op posities 1 en 3 staan. De selectie is protocolconform, maar eventuele positie-effecten zijn dus niet volledig gecounterbalanced.

## Once-only test en poorten

De test bevat exact 120 eindige samples:

| mean | p50 | p95 | p99 | min | max |
|---:|---:|---:|---:|---:|---:|
| 43.219529 | 43.154831 | **43.708455** | 44.228420 | 42.807072 | 44.783585 |

- Effectieve payloadbandbreedte bij p95: **22.265934 GB/s**.
- Marge onder de sterke 45-ms-poort: **1.291545 ms** (2.870%).
- Marge boven 21,627 GB/s: **0.638934 GB/s** (2.954%).
- Zelfs het langzaamste opgeslagen testsample is 0.216415 ms onder 45 ms.

Alle acht opgeslagen gates zijn exact herberekend als `true`: mismatchscalar, 120 samples, 65-ms/15-GB/s mechanismepoorten, 45-ms/21,627-GB/s sterke poorten, 48 registratieranges en lokale error/cleanup. `error=null` en de unregister-foutenlijst is leeg.

## Include- en protocolaudit

- Preregistratie-, evaluator-, result- en manifestprovenance matchen. De evaluatorhash is de huidige bronhash.
- De officiële lokale `cuda_pipeline.h` bestaat en is in de audit gehasht. De runner gebruikt expliciet het bundled CUDA-13-includepad.
- De kernel bevat `__pipeline_memcpy_async(...,16)`, commit, wait en 4-KiB-SMEM; geen `<stdint.h>`/`uintptr_t`-rest van D3.
- De geometrie sluit exact: 495 × 4.096 = 2.027.520 bytes per record en 237.600 × 4.096 = 973.209.600 bytes per token.
- Dit is geen TMA-tensormap en heeft geen verborgen fallbackarm.

Een provenancebeperking blijft: D5 importeert routes, stats, unregister en de verify-kernel uit D2, maar slaat de D2-modulehash niet in zijn eigen JSON op. De huidige D2-bron matcht wel exact haar bewaarde D2-evaluatorhash.

## Byte-evidencegrens

De verifier reconstrueerde de exacte correctnessroute en scande alle 480 records/973.209.600 geselecteerde bronbytes: nul structurele bronmismatches. De D5-uitvoer bewaart echter alleen `full_destination_mismatch_count: 0`, geen GPU-destinationhash of buffer. De tijdelijke volledige GPU-bestemming is daarom niet post-hoc CPU-only replaybaar.

## Full-bank- en claimgrens

Dit is een sterke **componentpass op slechts 307/512 experts per laag**: 29.877.534.720 bytes of 27,8256 GiB geregistreerd. `full_bank_pass=false` is correct; D5 heeft de 100%-bank niet uitgevoerd en herstelt D2's full-registrationfail niet.

De kernel verplaatst synthetische bytes van mapped host via `cp.async` naar SMEM en schrijft ze vervolgens naar een volledige HBM-oraclebuffer. Er is geen Q5 multiply/reductie, ERGV-integratie, TMA-descriptor, echte 80B-checkpoint, modelkwaliteit, dense shell, end-to-end tokens/s, page-telemetrie of endurance bewezen.
