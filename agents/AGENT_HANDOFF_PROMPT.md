# Startprompt voor een nieuwe agent

Kopieer alles onder de streep als eerste bericht in een nieuwe sessie.

---

Je neemt een werkende MoE-inferentieruntime over: **Nemotron 3.5 Lightning
30B-A3B NVFP4** die causaal draait op een **8 GiB RTX PRO 2000 Blackwell
laptop-GPU**, door de experts vanaf host te streamen. Werkdirectory:
`C:\Users\de_do\Documents\ChatGPT\New project`. Python:
`./.venv-nemotron/Scripts/python.exe`.

**⚠️ Modelpad — kritiek, lees dit voordat je iets meet.** Er staan twee
modellen in `models/`: `nemotron_3_5_lightning` (géén suffix) is bij
nameting **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4**, verkeerd gedownload en
misleidend hernoemd — de hele closed `treesweep200`-lijn (A1/D1/E1-E6/NERVF,
het bevroren `V36_DETERMINISTIC_ANCHOR.json`) draait daar nog steeds
standaard op. Het échte opdrachtdoel staat in `models/nemotron_3_5_lightning_v35`
— zet `LS_MODEL_DIR=nemotron_3_5_lightning_v35` (of gebruik `pro_research`,
dat dit al als default heeft). Volledige bewijsketen:
`agents/RESEARCH_NOTEBOOK.md`, blok 2026-08-16 "PRO V3 anchor-onderzoek".

**Nieuw sinds 2026-08-16: de `pro_research/`-addon.** Een geïsoleerde,
additieve experimentenpak (eigen branch `pro-research`, schrijft alleen naar
`pro_research/results/`, raakt nooit bestaande runtime-bestanden aan — ziet
`pro_research/README.md`). Huidige beste geverifieerde resultaat: **41,13
tok/s** (V4, full mode, 765 samples, alle poorten groen) — beter dan wat
hieronder over de oude (Nano-)lijn staat. Lees `agents/STATE_OF_THE_WORK.md`
z'n Lightning-tabel en `agents/RESEARCH_NOTEBOOK.md`'s 2026-08-16-blokken
vóór je verder graaft — daar staat ook waarom MTP-speculatief-decoderen
inmiddels gesloten is (57,51 vs 54,28 ms/token, gemeten) en wat de volgende
geprereg­istreerde stap is (`pro_research/PRO_V5_PREREGISTRATION.md`,
down_proj-kernels batchen — nog niet gebouwd).

## Lees eerst, in deze volgorde

1. `agents/STATE_OF_THE_WORK.md` — waar we staan, wat bewezen is, wat weerlegd
   is, hoeveel tok/s, en waar alles staat.
2. `agents/TODO.md` — de enige actieve takenlijst. Hier vink je af.
3. `agents/README.md` — de werkregels.
4. `agents/RESEARCH_NOTEBOOK.md` — het logboek; hier schrijf je per fase één blok
   bij, nieuwste bovenaan.

Verder graven kan in `reports/treesweep200/EXPERIMENT_REGISTRY.yaml` (31
experimenten met status en poorten) en `reports/nervf_nemotron/NERVF_NEMOTRON_FINAL_REPORT.md`.

## Werkregels — deze gelden hard

1. **Schrijfrechten**: alleen `reports/`, `scripts/`, `src/moe_lab/`, `tests/`
   binnen de `lightningstream_nemotron`-, `treesweep200`- en
   `nervf_nemotron`-namespaces, plus `agents/`, `models/nemotron_3_5_lightning*`
   en `docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md`. Al het andere is
   **read-only**: Codex werkt aan de 80B-lijn in `reports/streamq5_moe/`.
2. **Na elke fase** `scripts/lightningstream_nemotron/protected_manifest.py verify`
   draaien met `--baseline reports/lightningstream_nemotron/PROTECTED_80B_MANIFEST_BEFORE.json`.
   Eis: **0 modified / 0 removed**. "added" is je eigen werk.
3. **GPU delen** via `nvidia-smi --query-compute-apps`. **Nooit een proces
   killen.** Elke runner controleert dit zelf en stopt met exit 4.
4. **Preregistreer vóór je meet**: schrijf de poorten op in een `*_PREREGISTRATION_*.md`,
   dán de runner, dán een **aparte verifier** die alles herberekent zonder de
   runner te importeren, dán het rapport met een claim boundary.
5. **Verruim nooit een poort achteraf.** Een gefaalde poort is een resultaat.
6. **Eén variabele per meting.**
7. **Waardeer nooit een componentmeting op tot tok/s.** Kernelpercentages mogen
   niet worden opgeteld tot een tokenwinst — daar is een fysieke A/B voor nodig.
8. **Bouw een controle-arm die moet falen.** Slaagt hij, dan heeft je test geen
   onderscheidend vermogen en bewijst je hoofdpoort niets. Vier fasen misten een
   echte niet-determinismefout omdat dit ontbrak; fase A1 laat zien hoe het wel
   moet.
9. Rapporteer eerlijk. Een weerlegging netjes opgeschreven is net zoveel waard
   als een winst.

## Waar je aan werkt

Doel: **zoveel mogelijk tok/s, exact.** Nu 26,7 tok/s in een 512-token rollout,
29,5 bij ctx 0. Roofline-plafond van deze machine is 165 (ctx0) / 119 (lang), dus
er is hoofdruimte — de runtime draait op ~17% van zijn roofline. 50 en 100 tok/s
zijn niet uitgesloten; **1000 wel** (338,4 GB/s gemeten leesbandbreedte legt een
ondergrens van 6,05 ms per token).

De open lijst staat in `agents/TODO.md`. Bovenaan staat **E1 fase 2**: de echte
graph-resident token, budget **8,9 ms per token**, geblokkeerd op device-side
routing. De host-read-variant is al gesloten (V1: 6,7 vs 85,9 GB/s), dus dit
vraagt een nieuw ontwerp, geen herhaling.

Kijk vóór je begint in de tabel "Wat weerlegd is" in `STATE_OF_THE_WORK.md`.
Speculative decoding, gatherloze downflow en byte-reductie zijn langs meerdere
onafhankelijke paden dichtgemeten — herhaal die niet zonder een echt nieuw idee.

## Twee dingen die je makkelijk fout doet

- **Het juiste anker.** `V36_DETERMINISTIC_ANCHOR.json` voor al het nieuwe werk;
  `V35_GENERATION_ANCHOR.json` alleen voor metingen van vóór de A1-adoptie. Ze
  zijn **niet** bit-vergelijkbaar.
- **De attentiekernel wisselen.** Dat gaat nu via `rt.attn`, niet via
  `rt.k.attention_fp8_gqa`. Oude scripts die het oude pad overschrijven meten een
  nul-verschil.

Werk de todolijst af, schrijf tussentijds rapporten, en vink af in
`agents/TODO.md` plus een blok in `agents/RESEARCH_NOTEBOOK.md`.
