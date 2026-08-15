# GaugePack P9D-1 — audit van de P9B-pruningpremisse

Uitkomst: **p9b_pruning_noop_proven**.

De Qwen-forward gebruikt werkelijk de `ModuleList`-experts en hun gate/up/down-Parameters. Het defect zit in de mutatie: `weight[boolean_mask].zero_()` en `weight[:, boolean_mask].zero_()` werken op advanced-indexkopieën en schrijven niet terug naar de Parameter.

Op echte laag-0/expert-0-checkpointgewichten bleven alle drie SHA-256-hashes en alle aantallen niet-nulwaarden na de P9B-helper ongewijzigd; de expert-forward bleef bitexact. Een gecorrigeerde `masked_fill_`-mutatie nulde alle bedoelde waarden en veranderde 8,181 BF16-outputelementen in de vaste probe.

P9B en P9E verschillen op 2,018,499 van 2,359,296 opgeslagen indexposities en hebben nul identieke expert-laagmaskers, maar hun candidate CE, top-1, final-hidden error en alle 48 laag-errors zijn exact gelijk. Dat is nu mechanistisch verklaard.

Gevolg: de P9B-kwaliteitspass bewijst geen veilige 50%-pruning. P9C blijft een werkelijk uitgevoerde andere, destructieve compact/requantize-proef, maar kan P9B niet valideren. GaugePack P9D-1 is geblokkeerd tot P9B-R met een echte in-place mutatie validation én test passeert.
