# PORT80B D3/D4 — gecombineerde onafhankelijke CPU-only verificatie

**Verdict:** `all_four_negative_confirmed_with_evidence_limits`  
**GPU-context geopend:** nee  
**Alle replaybare reken-, selectie-, gate-, hash- en provenancechecks:** PASS

## Vier onafhankelijke conclusies

| Fase | Onafhankelijk bevestigd verdict | Fysieke data |
|---|---|---|
| D3 | compile-fail | NVRTC stopte op ontbrekende `stdint.h`; geen kernel, mismatch of timing |
| D3R | mapped-host physical negative | correctnessscalar 0; beste validation-p50 166,471 ms >65 ms, test bleef gesloten |
| D4 | compatibility fail | eerste native batchroute gaf illegal address; geen correctness/timing; 48 cleanupfouten |
| D4R | repaired native-batch physical negative | volledige validation/test; kleine winst, maar vier performancegates falen |

## D3 en D3R

D3's resultaat bevat exact de compilefout `cannot open source file "stdint.h"`, gevolgd door `Compilation terminated`. Er zijn geen validation-, test- of mismatchvelden. De unregister-foutenlijst is leeg. Dit is uitsluitend een compile-fail, geen negatieve kernelmeting.

D3R herhaalt dezelfde protocolgeometrie. Alle vier schedules hebben 24 eindige validation-samples:

| blocks | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms | diagnostische GB/s bij validation-p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 24 | 167.380914 | 166.471169 | 170.767634 | 171.785441 | 165.388611 | 172.062149 | 5.699028 |
| 1024 | 24 | 169.586653 | 168.915634 | 173.686597 | 177.026949 | 166.901093 | 177.887680 | 5.603251 |
| 2048 | 24 | 178.355494 | 177.454628 | 183.260664 | 185.307154 | 174.813248 | 185.916260 | 5.310521 |
| 4096 | 24 | 192.603945 | 191.808029 | 196.211764 | 198.003537 | 190.156418 | 198.526245 | 4.959996 |

De selectie is correct: 512 blocks heeft de laagste p50. Die p50 is 166.471169 ms, 101.471169 ms boven de 65-ms-openingspoort en 2.561095× de limiet. Er zou 60.954% latencyreductie nodig zijn. Daarom bleef de vooraf geregistreerde 120-sampletest dicht; test-p95 en testbandbreedte zijn formeel `null`, niet nul of een geschatte fail.

Alle 24 tokens en rotatie/omkeerorders kloppen; iedere van de vier werkelijk voorkomende orders staat zesmaal in de reeks. Registratie is 48×307 records, aliases zijn niet nul, `error=null` en unregisterfouten zijn leeg. Alle herberekende D3R-gates matchen de JSON.

## D4 en D4R

D4 vond de native `cudaMemcpyBatchAsync`-symbol en ABI-groottes 8/24, maar de eerste native route vergiftigde de context met `cudaErrorIllegalAddress`. Er zijn geen correctness-, validation-, test- of gatevelden. Alle 48 unregistercalls rapporteerden dezelfde illegal-addressfout. Dit is een valide compatibiliteitsfail, geen snelheidstest.

D4R gebruikt voor native descriptors de niet-nulle `devicePointer`-aliases, terwijl de ordinary arm CPU-hostpointers behoudt. De drie validationarmen zijn onafhankelijk herberekend:

| arm | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary480 | 24 | 51.966123 | 51.893553 | 52.316919 | 52.414771 | 51.754208 | 52.439007 |
| batch48x10 | 24 | 50.516163 | 50.474319 | 50.836029 | 50.922789 | 50.301441 | 50.943264 |
| batch1x480 | 24 | 50.551398 | 50.535183 | 50.705386 | 50.734175 | 50.354782 | 50.742462 |

`batch48x10` is correct geselecteerd. De validation-p50-/p95-ratio's versus ordinary zijn 0.972651067 en 0.971693863; dit zijn slechts 1.028118× en 1.029131× snelheidsfactoren, niet de vereiste ratio ≤0,90.

De once-onlytest bevat exact 120 eindige samples: mean 50.486987, p50 50.487761, p95 50.694517, p99 50.783813, min 50.265503, max 50.818783 ms. De p95 ligt 5.694517 ms boven 45 ms; 11.233% reductie ontbreekt. Bandbreedte is 19.197532 GB/s, 2.429468 GB/s onder de gate.

De vier performancefails zijn: test-p95, p95-bandwidth, validation-p50-ratio en validation-p95-ratio. Native symbol/ABI, 3× mismatchscalar nul, 120 samples, registratie en lokale error/unregisterpoorten passeren. `error=null`; unregisterfouten zijn leeg.

## Repairs en provenance

- D3R voldoet in de huidige bron aan de vastgelegde repair: geen `<stdint.h>`/`uintptr_t`, wel directe `unsigned long long`-pointercast. Protocolvelden zijn gelijk aan D3.
- D4R voldoet in de huidige bron: native descriptors gebruiken devicealiases, ordinary gebruikt hostpointers, aliases worden op nul gecontroleerd. Protocolvelden zijn gelijk aan D4.
- De huidige runners matchen exact de opgeslagen D3R/D4R-evaluatorhashes.
- **Beperking:** de oorspronkelijke D3/D4-runnerbronnen onder hashes `214d33d7c3d11264612492db9b3fdb91c793e9a8d69986e23fd04d6133498085` en `96da20584f5d263f04e1671128424ec75f816cd953b4c49904581afe1bdb3fc7` zijn overschreven. Een exacte source-diff kan daarom niet worden gecontroleerd; alleen de huidige repair, gelijke protocols en bewaarde failureartefacten.
- De fysieke bank is opnieuw volledig CPU-side gehasht: `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`.

## Byte-evidencegrens

De correctnessroutes voor D3R-token 49.999 en D4R-token 69.999 zijn onafhankelijk gereconstrueerd. Voor beide zijn alle 480 records en alle 973.209.600 geselecteerde bronbytes gescand: nul structurele bronmismatches.

D3R en D4R bewaren echter alleen GPU-mismatchscalars, geen destinationhashes/buffers. De tijdelijke GPU-bestemmingen — en bij D4R de drie afzonderlijke armuitkomsten — kunnen daarom niet post-hoc CPU-only worden hervergeleken. Dat beperkt de reproduceerbaarheid van correctness, maar niet de negatieve timingverdicts.

Geen van deze fasen bewijst Q5-aritmetiek, een echte 80B-port, kwaliteit, dense-shell-timing, end-to-end tokens/s, full-bankcapaciteit of endurance.
