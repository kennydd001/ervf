# P9C — naïeve compacte Q5-regroepering gesloten

## Uitkomst

De fysiek aantrekkelijke representatie met gate/up `[384,2048]` en down
`[2048,384]`, opnieuw per 128 aaneengesloten down-gewichten gekwantiseerd,
faalde de validation-poorten:

- relatieve cross-entropy: **+48,027%**;
- top-1-overeenkomst: **60,472%**;
- eind-hidden relatieve L2: **0,4866**.

De test-split blijft volgens preregistratie ongeopend. Het negatieve resultaat
is informatief: willekeurig gekozen neuronen mogen niet zonder meer nieuwe
quantisatiegroepen vormen. Alleen een groepsbehoudende compacte codering of een
nieuw gekalibreerde selectie blijft als fysieke vervolgrichting open.
