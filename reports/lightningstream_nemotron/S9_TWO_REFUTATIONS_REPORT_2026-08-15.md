# S9 — twee hypotheses weerlegd, en waar de MoE-tijd wél zit

Datum: 2026-08-15
Verdict: **H-S9 (launch-batching) en H-S9b (blokgrootte) allebei weerlegd vóór er iets gebouwd werd. De GEMV's verklaren maar ~23% van de MoE-term; de rest zit in het S5-gatherpad.**
Terminal state: `s9_launch_and_blocksize_refuted_gather_suspected`

## H-S9 — launch-overhead: weerlegd

S8 vermoedde dat ~550 launches/token de 39,523 ms MoE-term verklaarden. Gemeten
met een lege kernel op deze GPU:

| grootheid | waarde |
|---|---:|
| lege launch | 7,020 µs |
| launch met grid 1856 | 7,956 µs |
| fused-calls per MoE-laag | 14 |
| geschatte launches/token (MoE) | 644 |
| **launch-overhead schatting** | **5,124 ms** |
| **aandeel van de MoE-term** | **13,0%** |

Batchen van zes experts in één launch zou hooguit ~4 ms opleveren tegen de
38 ms kloof. **Kernel-bouwproject afgeblazen op een probe van tien regels** —
precies waarvoor die discipline bedoeld is.

## H-S9b — blokgrootte / reductieboom: weerlegd

Vermoeden: 1856 blocks die elk 1344 B lezen zijn reductie-gebonden, dus een
kleiner blok zou sneller zijn. Microbenchmark op één echte `up_proj`:

| block | µs | effectief |
|---:|---:|---:|
| 32 | 113,64 | 24,7 GB/s |
| 64 | 56,26 | 49,9 GB/s |
| 128 | 36,94 | 76,0 GB/s |
| **256 (huidig)** | **32,45** | **86,5 GB/s** |
| 512 | 39,18 | 71,6 GB/s |

**256 is al optimaal**; kleiner is monotoon slechter. De hypothese was precies
verkeerd om.

## Wat de meting wél oplevert

Bij 32,45 µs per `up_proj` is een expert (up + down) ruwweg 65 µs. Over 138
experts is dat **~9,0 ms** — tegen een gemeten MoE-term van **39,5 ms**. De
GEMV's verklaren dus maar ~23%.

De resterende ~30 ms zit in de rest van het S5-pad: `panel_scan`,
`gather_down_sparse`, de accumulatie, de router en de shared expert.

**Concreet vermoeden voor de volgende fase, nog niet gemeten:** de cache is sinds
S5 **up-only**. Bij 80,4% hitrate heeft dus 80% van de experts zijn up-helft op
device — maar **100% van de experts haalt down elke keer opnieuw via
`gather_down_sparse` uit mapped host-geheugen**, gemeten op 25,05 GB/s tegenover
~250 GB/s device. Een hit bespaart daarmee alleen de up-helft.

Dat is een testbare vraag met één variabele: **volledig-record caching op halve
capaciteit** versus **up-only op volle capaciteit**. S5 werd gebouwd toen de
premisse was dat PCIe-miss-bytes domineerden; S8 heeft die premisse weerlegd,
dus de afweging moet opnieuw.

## Meetnotitie

De blokgrootte-probe rapporteerde `outputs_identical_across_blocks: False` met
`rel_l2 ≈ 2e-07`. Dat is geen fout: een andere blokgrootte geeft een andere
reductieboom. Mijn gelijkheidscheck was te streng geformuleerd — hij vergeleek
tegen block 32 in plaats van tegen een tolerantie. De waarde 2e-07 ligt binnen
de gebruikelijke 1e-5-poort.

## Claim boundary

Componentmetingen op deze GPU: launchkosten met een lege kernel, GEMV-tijd op
één echte matrix. De per-token schattingen vermenigvuldigen die met getelde
launches respectievelijk expert-calls — het zijn schattingen van een grens, geen
tokenmetingen. Geen tok/s- of kwaliteitsclaim.

## Artefacten

`s9a_launch_probe.py` · `s9a_launch_probe.json` ·
`s9b_blocksize_probe.py` · `s9b_blocksize_probe.json`
