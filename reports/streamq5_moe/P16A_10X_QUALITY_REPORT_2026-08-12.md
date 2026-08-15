# P16A — 10× grotere Q5-kwaliteitsaudit

## Uitkomst

De vaste uniforme Q5-expert/INT8-trunkkandidaat is full-depth geëvalueerd op
100 contexten uit vijf domeinen, samen 12.700 volgende-tokenlabels:

- teacher CE: 1,885518;
- kandidaat CE: 1,912890;
- relatieve toename: **+1,4517%**;
- top-1-overeenkomst: **92,9528%**;
- contextbootstrap, 10.000 trekkingen: **[+1,1542%, +1,7619%]** (95%).

Alle vooraf vastgelegde poorten passeren. Dit vermindert de onzekerheid van de
eerdere 1.270-labeltests sterk.

## Grens

De data waren eerder gebruikt voor route- en GPTQ-calibratieonderzoek. Deze run
is dus een grote corroboratieve audit, geen onaangeraakte publieke benchmark en
geen onafhankelijke externe reproductie.
