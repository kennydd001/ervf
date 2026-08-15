# E5 — GEMV roofline recovery via ERVF: 1,660x gewogen, bitexact, poort net niet

Datum: 2026-08-15 · Registry `TREESWEEP200`
Verdict: **Alle vier de NVFP4-shapes die de runtime gebruikt worden 1,62 tot 1,78x sneller en zijn stuk voor stuk bit-identiek. Gewogen komt de suite van 77,1 op 127,9 GB/s (1,660x). De primaire poort vroeg >=140 GB/s en wordt net niet gehaald; de regressiepoort wordt ruim gehaald.**
Terminal state: `e5_ervf_16x_gain_all_shapes_exact_suite_gate_missed`

## Mechanisme

Niet een nieuwe kernel voor E5, maar **ERVF** uit de NERVF-lijn: 16-lane
subwarps, 16 rijen per block van 256 threads, per lane gescheiden virtuele
accumulatoren, en de referentie-reductieboom exact gereconstrueerd. E5 meet wat
die op **elke** shape doet die `gemv_nvfp4_rows` in de echte lus krijgt, gewogen
naar hoe vaak ze per token wordt aangeroepen, alle armen L2-koud.

## Suite

| shape | afmeting | aanroepen/token | basis | ERVF | speedup | bitexact |
|---|---|---:|---:|---:|---:|:--:|
| routed_up | 1856x2688 | 138 | 75,9 GB/s | **123,0** | 1,621x | ✅ |
| shared_up | 3712x2688 | 23 | 79,9 | **142,5** | 1,784x | ✅ |
| shared_down | 2688x3712 | 23 | 78,0 | **128,3** | 1,646x | ✅ |
| lm_head | 131072x2688 | 1 | 77,2 | **129,3** | 1,676x | ✅ |
| **gewogen** | | | **77,1** | **127,9** | **1,660x** | ✅ |

| poort | eis | gemeten | |
|---|---|---:|:--:|
| `weighted_suite_bandwidth_ge_140_gb_s` | >= 140 GB/s | **127,9** | ❌ |
| `no_critical_shape_regression_gt_5pct` | geen shape >5% trager | slechtste **1,621x** | ✅ |
| `integrated_token_improvement_ge_8pct` | >= 8% | 9,8% @ctx0, 8,1% @262100 (NERVF-3) | ✅ |
| strong `>= 170 GB/s` | | 127,9 | ❌ |
| exactheid, alle shapes | bit-identiek | 4 van 4 | ✅ |

## Wat dit wel en niet zegt

De poort wordt **niet verruimd**. 127,9 tegen 140 is 8,6% tekort.

Twee dingen die er wel staan. Ten eerste: er is **geen enkele shape die
achteruitgaat** — dat is de gate die een integratie blokkeert, en die is ruim
gehaald met een slechtste geval van 1,621x. Ten tweede: de winst is
**bitexact op elke shape**, dus adoptie kost geen enkele semantiekconcessie.

Een meetnotitie voor de eerlijkheid: NERVF-2 mat op `routed_up` alleen 140,8 GB/s,
hier 123,0. Dezelfde kernel, andere meetcondities (kortere runs, kleinere pool,
en deze shape draait hier als eerste van vier). Ik rapporteer het lagere getal,
want dat is de suite-meting waar de poort op slaat.

Het gat naar 140 zit niet in ERVF maar in wat eronder ligt: N1 mat dat 23,7% van
een token uitgifte-overhead is en NERVF-1 dat na de decode nog 46% in reductie en
synchronisatie zat — ERVF haalt daar het grootste deel van weg, en wat overblijft
vraagt de graph-resident uitvoering van E1.

## Claim boundary

Microbenchmarks op de echte NVFP4-tensors van dit checkpoint, elke shape die
`gemv_nvfp4_rows` in de runtime krijgt, alle armen door een L2-koude pool. Het
gewogen cijfer weegt elke shape naar zijn aanroepfrequentie per token en is een
**kernel**-bandbreedte, geen tokentijd en geen doorvoerresultaat. Exactheid is
per shape tegen de productiekernel gecontroleerd. De 9,8%/8,1%
token-verbetering komt uit NERVF-3 en is daar end-to-end gemeten.

## Artefacten

`scripts/treesweep200/e5_gemv_roofline_suite.py` · `E5_GEMV_ROOFLINE_SUITE.json`
