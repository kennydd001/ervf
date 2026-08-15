# Synthese — waar 50 tok/s bij lange context fysiek strandt

Datum: 2026-08-15
Status: eindsynthese van de LIGHTNINGSTREAM_NEMOTRON-lijn na S13 en S14.
Geen nieuwe meting; dit document legt de grens vast die uit de geverifieerde
fasen volgt. Elke bouwsteen staat in zijn eigen rapport met eigen verifier.

## 1. Waar we staan (gemeten, geverifieerd)

Nemotron 3.5 Lightning 30B-A3B NVFP4, één 8 GiB laptop-GPU, batch 1, greedy:

| context | tok/s | ms/token |
|---:|---:|---:|
| 0 | 27,7 | 36,1 |
| 32.768 | 26,2 | 38,2 |
| 131.072 | 21,7 | 46,1 |
| 262.100 | 18,4 | 54,3 |

## 2. Elke as is gemeten en afgesloten

| as | fase | uitkomst |
|---|---|---|
| routevoorspelling / prefetch | S1 | recall@24 = 0,724 < 0,80 — weerlegd |
| ReLU²-sparsity (kolom-selectieve down_proj) | S2, S5 | gebouwd, exact, +4,1% — geen gamechanger |
| lossless/delta-codering van experts | S3 | entropie 3,967/4 bits — weerlegd |
| speculatief decoderen (MTP bestaat wél) | S10A, **S13** | acceptatie A=2,114, maar expert-unie 19,5/128 bij W=5 → per gecommitteerde token **meer** MoE-bytes (6,27 vs 6,00 records) — bouwpoort gefaald, niet gebouwd |
| volledig-record caching | S11 | −4,84% bij gelijke bytes — weerlegd |
| transfergebondenheid van de MoE-term | S8, S11 | 2,9× meer PCIe kost 4,8% — niet transfergebonden |
| één dominante kernel in de MoE-term | S12 | bestaat niet; marginalen dekken 15,5 van 39,5 ms |
| waar de rest zit | **S14** | volledig toegeschreven: 27,7 ms echte MoE-streamtijd + ~12 ms afdrain-artefact van vreemd werk bij de readback-sync |

De LIGHTNINGFLASH_50-hypothesen uit `info/SWEEPSPEC_50_PACK_2026-08-14`
(tree-verificatie, draft-forest, overlap) erven allemaal de S13-voorwaarde en
zijn voor dit model gesloten.

## 3. De fysieke grens bij 262K (vloer-aritmetiek op gemeten componenten)

50 tok/s vereist 20 ms/token. De gemeten vloeren, semantiek ongewijzigd:

| term | vloer (ms) | waarom het een vloer is |
|---|---:|---|
| attention, 6 lagen | 3,3 | KV eenmaal lezen: 805 MB FP8-KV per token à 244,8 GB/s (S7-mechanismecheck) |
| expert-werk, 23 lagen | ~16 | up-GEMV + sparse down-gather per geraakte expert (S9/S14) |
| Mamba, 23 lagen | ~8,3 | alleen geïsoleerd gemeten (S8); in-lus ligt hij lager |
| lm_head | 2,1 | vocab-GEMV |
| **som** | **≈ 30** | → **≈ 33 tok/s bij 262K** |

Onder die grens komen vereist minder KV-bytes of minder expert-bytes per
gecommitteerde token — allebei semantiekveranderingen (andere kwaliteit), die de
registry expliciet verbiedt. **50 tok/s bij 262K ligt buiten de gemeten fysica
van dit model op deze GPU.** Bij kortere context schuift de grens omhoog
(attention ~0 bij ctx 0; daar is de MoE-term van 27,7 ms de bindende vloer,
~36 tok/s als al het andere gratis zou zijn).

## 4. Wat wél nog kan (engineering, geen nieuwe wetenschap)

Uit S14's segmenten, bij 262K, zonder semantiek te raken:

- `host_gap` 4,7 ms — GPU staat stil terwijl de host readback-afhandeling, LRU
  en copy-issue doet. Weg te halen door de boekhouding asynchroon of op device
  te doen.
- `route` 3,5 ms — 344 kFLOP in ~10 kernel-launches per laag; te fuseren tot één.
- in-lus `up`+`down` 15,0 ms tegen ~9,0 ms microbenchmark — L2/gather-effecten.
- attention 18,6 → richting 3,3 ms vloer — de GQA-kernel is compute-bound op
  shuffles (S7-analyse); de vloer is gemeten bereikbaar in principe.

Samen goed voor richting **25–30 tok/s bij 262K**. Dat is de eerlijke
eindstreep van deze lijn: meetbaar, bouwbaar, maar het is engineering tegen
bekende vloeren — en het haalt 50 niet.

## 5. Claim boundary

Dit document voegt geen meting toe; het combineert uitsluitend afzonderlijk
geverifieerde faseresultaten (S1–S14) en labelt elke afgeleide grootheid als
rekenwerk. De grens in §3 geldt voor dit checkpoint, deze runtime-semantiek,
deze GPU, batch 1. Zij zegt niets over andere modellen, andere hardware, of
over semantiekveranderende methoden (kwantisatie van KV onder FP8, pruning,
surrogaat-modellen — allemaal buiten de opdracht of verboden).
