# `agents/` — de werkmap voor iedereen die aan deze runtime werkt

Dit is het startpunt. Drie bestanden, elk met één taak:

| bestand | waarvoor |
|---|---|
| [`STATE_OF_THE_WORK.md`](STATE_OF_THE_WORK.md) | **Lees dit eerst.** Waar we staan, wat bewezen is, wat weerlegd is, hoeveel tok/s. |
| [`TODO.md`](TODO.md) | De enige actieve takenlijst. Afvinken doe je hier. |
| [`RESEARCH_NOTEBOOK.md`](RESEARCH_NOTEBOOK.md) | Chronologisch logboek. Elke fase schrijft hier één blok bij, nieuwste bovenaan. |

Voor een agent die nieuw begint staat de volledige startprompt in
[`AGENT_HANDOFF_PROMPT.md`](AGENT_HANDOFF_PROMPT.md).

## De werkregels, kort

1. **Schrijfrechten**: alleen `reports/`, `scripts/`, `src/moe_lab/`, `tests/`
   onder de `lightningstream_nemotron`-, `treesweep200`- en
   `nervf_nemotron`-namespaces, plus `agents/`, `models/nemotron_3_5_lightning*`
   en `docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md`. Al het andere is
   **read-only** — Codex werkt aan de 80B-lijn in `reports/streamq5_moe/`.
2. **Na elke fase** `protected_manifest.py verify` draaien. De eis is
   **0 modified / 0 removed**; "added" is ons eigen werk.
3. **GPU delen** via `nvidia-smi --query-compute-apps`. **Nooit** een proces
   killen. Elke runner controleert dit zelf en stopt met exit 4.
4. **Preregistratie vóór uitvoering**: poorten opschrijven, dán meten, dán een
   *aparte* verifier die alles herberekent zonder de runner te importeren.
5. **Poorten worden nooit achteraf verruimd.** Een gefaalde poort is een
   resultaat, geen probleem.
6. **Eén variabele per meting.**
7. **Nooit een componentmeting opwaarderen tot tok/s.** Percentages van losse
   kernels mogen niet worden opgeteld tot een tokenwinst — daar is een fysieke
   A/B voor nodig.
8. Bouw een controle-arm in die **moet falen**. Als hij slaagt, heeft je test
   geen onderscheidend vermogen en bewijst je hoofdpoort niets. (A1 is hier het
   voorbeeld van; vier eerdere fasen misten een echte fout doordat dit ontbrak.)
