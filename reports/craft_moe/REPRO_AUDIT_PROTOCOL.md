# CRAFT-MoE reproducibility-auditprotocol

Vastgelegd op 2026-08-10 nadat alle niet-geblokkeerde technische hypothesen
waren afgesloten en vóór uitvoering van `verify_all_gates.py`. Deze audit mag
geen technische gate wijzigen, geen testgestuurde variant openen en geen
negatieve uitkomst promoveren.

## Scope

De audit controleert alle bewaarde H1–H4-, H6–H8- en H10-resultaten, inclusief
positieve tussenstappen, harde stops, inconclusieve screens en gecorrigeerde
append-only audits. H5, H9 en de packed runtime blijven dependency-geblokkeerd.

Verplichte controles:

1. schema, status, model- en datasetrevision;
2. herberekening van iedere gerapporteerde gate uit de numerieke metriekvelden;
3. exacte original-, route- en finite-controls;
4. validatie/testscheiding en expliciete afwezigheid van testselectie;
5. deterministische byte-, union-, miss- en gap-accounting;
6. sequence-blockbootstrap en onafhankelijke herberekening voor H8 en H10;
7. alle vijf vaste H4-seeds en het ontbreken van ongeopende trained selectors;
8. behoud van H4-v1 en H2-audit-v1 als falende artefacten;
9. SHA-256 van resultaat-, capture-, support- en auditbestanden;
10. benchmarkdiscipline: warmup en herhalingen worden gecontroleerd; ontbrekende
    thermal-steady-state-telemetrie is een auditwaarschuwing en verhindert een
    wall-clockclaim.

H7 wordt opnieuw uit de per-token-KL-series opgebouwd. H8 wordt opnieuw uit de
twaalf validatieconfiguraties geselecteerd. Voor H10 wordt de validation-argmin
opnieuw rechtstreeks uit het lossless `8×720×256`-sweepbestand berekend. De
twee H8/H10-blockbootstraps worden met dezelfde vooraf vaste seeds maar via een
afzonderlijke auditimplementatie gereproduceerd.

## Beslisregel

`verify_all_gates.py` eindigt met een niet-nul exitcode zodra één verplichte
controle faalt. Waarschuwingen mogen alleen beperkingen beschrijven die geen
gerapporteerde gate of claim ongeldig maken. De eerste volledige audit schrijft
een append-only hashmanifest naar `repro_audit.json`; latere `--check-only`-runs
moeten exact tegen dat manifest controleren en schrijven niets.

