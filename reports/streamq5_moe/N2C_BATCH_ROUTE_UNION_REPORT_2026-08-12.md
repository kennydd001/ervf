# N2C — batch/route-union-sweep

Datum: 2026-08-12. Status: **fysieke hypothese gesloten**.

## Uitkomst

De temporal-Q8 plus sparse-temporal-Q5 batchconstructie is bitexact voor alle
voltooide groottes, maar is trager dan sequentiële N1B-Q5 plus ERVF-16-Q8.
Geen testpartition werd geopend, conform de vooraf geregistreerde
validationpoort van `<=0,98`.

| S | route-unie p50 / p95 | Q5 p50-ratio | gecombineerd p50-ratio | gecombineerd p95-ratio | besluit |
|---:|---:|---:|---:|---:|---|
| 2 | 12 / 15 | 3,4021 | 2,0120 | 2,0086 | test gesloten |
| 4 | 19 / 26 | 3,1671 | 1,7939 | 1,7974 | test gesloten |
| 8 | 27 / 39 | 3,3584 | 2,0386 | 1,0191 | test gesloten |
| 16 | — | — | — | — | resource-spill-timeout |

Bitexactheidsomvang:

- S=2: Q8 1.004.288 en Q5 13.762.560 elementen, nul verschillen;
- S=4: Q8 2.008.576 en Q5 27.525.120 elementen, nul verschillen;
- S=8: Q8 4.017.152 en Q5 55.050.240 elementen, nul verschillen.

Een tweede run reproduceerde de eerste-run-p50-ratio's nauw:

- eerste run gecombineerd: 2,0176 / 1,8442 / 2,0472 voor S=2/4/8;
- bewaarde run: 2,0120 / 1,7939 / 2,0386.

## Interpretatie

De echte routerlijsten overlappen wel, maar de gekozen algemene sparse kernel
houdt per union-expert een variabel aantal actieve tokens en ranks bij. De
dynamische MAC-lus en de tokenaccumulatorstaat kosten op deze GPU veel meer dan
de uitgespaarde Q5-weightloads. N2A's positieve same-eight-expert-orakel
generaliseert dus niet naar deze algemene route-unionimplementatie.

Dit falsificeert niet ieder mogelijk batchkernelontwerp. Een gespecialiseerde
codegenvariant per exact membershipmasker, token-major tiling of een kernel die
Q5-decode over meerdere warps verdeelt, heeft andere uitvoeringsgeometrie. Wat
wel gesloten is: deze algemene per-union-expert dynamic-active-list-kernel voor
S=2/4/8; S=16 is in dezelfde vorm praktisch onbruikbaar.

## Thermische en methodologische grens

De laptop-GPU bereikte tijdens de oorspronkelijke lange sweep 85–86 °C en
ongeveer 1,7–2,0 GHz. Referentie en kandidaat zijn binnen ieder S afwisselend
AB/BA gemeten; de primaire per-S-ratios blijven daardoor bruikbaar. Absolute
tijden tussen verschillende S-waarden worden niet als een zuivere schaalcurve
of winnaarvergelijking geclaimd. S=16 is afzonderlijk beschreven in
`N2C_S16_RESOURCE_ABORT_2026-08-12.md`.

## Auditspoor

- preregistratie: `N2C_BATCH_ROUTE_UNION_SWEEP_PREREGISTRATION.md`;
- evaluator: `scripts/streamq5_moe/run_n2c_batch_route_union_sweep.py`;
- ruwe events: `n2c_batch_route_union_sweep.json`;
- verifier: `scripts/streamq5_moe/verify_n2c_batch_route_union.py`;
- onafhankelijke verificatie: `n2c_batch_route_union_verification.json`.

Claimgrens: fysieke residentiële componenttest op echte P4D-routerpatronen;
geen causale batchbeschikbaarheid, kwaliteit, end-to-end of externe SOTA-claim.
