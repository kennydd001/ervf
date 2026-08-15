# P9A vaste gemengde Q4/Q5-laagprecisie — preregistratie

Datum: 2026-08-12

## Hypothese

Een vooraf vastgelegde subset van twaalf robuuste routed-MoE-lagen kan van Q5
naar Q4 zonder de full-depth 2%-kwaliteitspoort te breken. Daarmee daalt de
expertbank/actieve expertbytes met circa 5% ten opzichte van uniform Q5.

## Selectie zonder nieuwe sweep

Uitsluitend de reeds geopende STREAMQ4-validation-laagcurven zijn gebruikt.
De kandidaat is éénmalig vastgezet op lagen:

`4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17`.

Alle andere lagen blijven Q5. Beide gebruiken group-128 symmetrische RTN,
FP32-codekeuze en een naar BF16 afgeronde opgeslagen schaal. De volledige
trunk en LM-head blijven INT8 zoals in P0C. Er volgt geen tweede subset als
deze faalt.

## Evaluatie en poorten

- P0C validation en daarna één test, ieder 1.270 labels en vijf domeinen;
- validation opent test bij relatieve CE `<=2,5%`, top-1 `>=90%` en 48 lagen;
- definitieve pass vereist validation én test CE `<=2,0%` en top-1 `>=90%`;
- een kwaliteitspass opent pas de fysieke mixed-bank- en kernelmeting.

