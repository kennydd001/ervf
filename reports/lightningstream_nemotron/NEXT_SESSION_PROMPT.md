# Startprompt voor de volgende sessie

Je neemt een werkende MoE-runtime over op een 8 GiB laptop-GPU. Lees eerst
`reports/lightningstream_nemotron/HANDOVER_TO_KIMI_2026-08-15.md` en
`S10_MTP_SPECULATIVE_PREREGISTRATION_2026-08-15.md`. Daarna één taak.

## Stand

Model: `models/nemotron_3_5_lightning_v35` (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).
Draaien met `LS_MODEL_DIR=nemotron_3_5_lightning_v35`.
Runner: `scripts/lightningstream_nemotron/n7b_cached_decode.py --capacity 72 --embed-on-host`

| context | tok/s |
|---:|---:|
| 0 | 27,743 |
| 32.768 | 26,200 |
| 131.072 | 21,699 |
| 262.100 | 18,424 |

Alle vier opdracht-poorten gehaald. Doel van de gebruiker: **50 tok/s bij lange
context**. Dat vereist ~20 ms/token; gemeten componenten (MoE-GEMV's 9,0 ms +
Mamba 8,3 ms + attention 18,6 ms geïsoleerd + lm_head 2,1 ms) sluiten dat langs
deze weg uit. Speculatief decoderen is de enige as die nog open ligt: het
verlaagt **bytes per token** in plaats van tijd per byte.

## Je taak: S10 stap 1 — acceptatiegraad meten

Bouw **geen** speculatieve lus. Meet eerst `A`, het aantal geaccepteerde
draft-tokens per sweep. De poort beslist of er ooit gebouwd wordt.

**Poort G-S10-1: gemiddelde `A` ≥ 1,5 over ≥200 stappen en 3 prompts.**
Daaronder gaat S10 dicht zonder bouw.

Methode: laat het MTP-blok `D=4` tokens voorstellen, laat het hoofdmodel greedy
de werkelijke tokens produceren, tel de match tot het eerste verschil. Twee
losse forwards, geen gedeelde staat.

### De MTP-wiring ligt vast (S10-A0, `s10a0_mtp_structure.json`)

```
h_mtp = eh_proj( concat( enorm(embed[token]), hnorm(h_prev) ) )   # [2688,5376]
      -> mtp.layers.0.mixer  q/k/v/o  (4096/256/256/4096, eigen KV)
      -> mtp.layers.1        gate top-6 + 128 BF16-experts + shared
      -> final_layernorm -> BACKBONE lm_head -> draft-logits
```

Belangrijk: **geen eigen embedding, geen eigen lm_head** — hergebruikt die van
de backbone. Alle benodigde kernels bestaan al (`gemv_bf16`, `rmsnorm_bf16w`,
GQA-attention, router). MTP-experts zijn **BF16**, dus `gemv_bf16` gebruiken,
niet de NVFP4-fused GEMV.

Skelet is maar 116 MB; de 2,49 GiB zit in de 128 experts. Voor de meting hoeven
die niet resident — zes per draft-token streamen volstaat (~236 MB totaal).

## Niet opnieuw proberen — gemeten en weerlegd

| hypothese | uitkomst |
|---|---|
| minder transcendentals in online softmax | 13,225 → 12,404 tok/s |
| meer flash-decode splits (256→1024) | 13,225 → 12,020 |
| launch-batching van experts | launch-overhead is 13% van de MoE-term |
| kleinere GEMV-blokken | 256 al optimaal, kleiner monotoon slechter |
| route-voorspelling (Kimi S1) | recall@24 = 0,724 < 0,80 |
| lossless codering / expert-delta (Kimi S3) | entropie 3,967/4 bits |
| cross-layer expert-prefetch | causaal onmogelijk |

Zie ook `forbidden_hypotheses` in `EXPERIMENT_REGISTRY.yaml` (pruning, low-rank
surrogaten, Q2-semantiek).

## Openstaand, niet opgelost

De cache is sinds S5 **up-only**: bij 80,4% hitrate haalt 100% van de experts
`down` elke keer uit mapped host op 25 GB/s tegen ~250 GB/s device. Een hit
bespaart dus alleen de up-helft. Testbaar met één variabele: volledig-record
caching op halve capaciteit versus up-only op volle. `enable_cache` en
`_moe_cached` zijn daarvoor verweven met het panel-major gatherpad en moeten
ontvlochten worden.

## Werkregels

- Schrijf alleen in `reports/`, `scripts/`, `src/moe_lab/`, `tests/` onder
  `lightningstream_nemotron/`, plus `models/nemotron_3_5_lightning*` en
  `docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md`.
- Alles daarbuiten read-only. Codex werkt aan de 80B-lijn in
  `reports/streamq5_moe/`. Draai `protected_manifest.py verify` na elke fase:
  **0 modified / 0 removed** is de eis; added is hun werk.
- GPU delen via `nvidia-smi --query-compute-apps`; nooit een proces killen. De
  Intel Arc iGPU is hun experiment.
- Preregistratie mét poorten vóór uitvoering, dan runner, dan een **aparte**
  verifier die alles herberekent zonder de runner te importeren, dan rapport met
  claim boundary. Poorten nooit verruimen na het zien van een resultaat.
- **Eén variabele per meting.** Ik heb daar zelf tegen gezondigd en het kostte
  een extra ronde.
- Nooit een componentmeting opwaarderen naar tok/s.
- **G-S10-C1 als er ooit gebouwd wordt:** de geaccepteerde tokenreeks moet
  identiek zijn aan de niet-speculatieve generatie. Speculatief decoderen dat de
  uitvoer verandert is een fout, geen afweging.

Reproduceer eerst de huidige meting. Haal je die niet, dan ligt het aan de
omgeving en niet aan je idee.
