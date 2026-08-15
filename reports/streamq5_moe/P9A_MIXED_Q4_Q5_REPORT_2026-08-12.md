# P9A — vaste gemengde Q4/Q5-laagprecisie

## Uitkomst

Twaalf vooraf gekozen robuustere lagen gebruikten Q4; de overige 36 Q5. Dat
reduceert de geprojecteerde expert-codebytes met 5% tegenover uniform Q5.

| split | relatieve CE-toename | top-1-overeenkomst |
|---|---:|---:|
| validation | +1,477% | 91,260% |
| test | −0,211% | 91,811% |

De kwaliteitspoorten passeren. Een fysieke mixed-bank/kernel is nog niet
gebouwd, zodat hier geen snelheidsclaim uit volgt.
