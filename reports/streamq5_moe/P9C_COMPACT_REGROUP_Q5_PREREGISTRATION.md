# P9C preregistratie — fysiek compacte regrouped-Q5 semantiek

Datum: 2026-08-12. Status bij vastlegging: output ongeopend.

## Vraag

P9B bewees kwaliteit nadat niet-geselecteerde kanalen in matrices van breedte
768 op nul waren gezet. Een echte compacte kernel bewaart slechts 384 kanalen;
voor de down-projectie veranderen daardoor de Q5-groepen. P9C test precies die
nog onbewezen semantiek vóór een bank van meerdere GiB wordt gebouwd.

## Kandidaat

- gebruik zonder wijziging de in P9B-validation verzegelde 384 indices per
  expert en laag;
- slice gate/up naar `[384, 2048]` en down naar `[2048, 384]`;
- kwantiseer die compacte matrices opnieuw met de vaste Q5/group-128-regel;
- scatter de gedequantiseerde compacte gewichten tijdelijk in nulmatrices van
  de oorspronkelijke vorm, uitsluitend om de bestaande model-evaluator te
  kunnen gebruiken;
- trunk en LM-head blijven INT8 zoals in P0C/P9B.

## Poorten

Validation opent test bij relatieve CE `<=2,5%`, top-1-overeenkomst `>=90%`,
alle 48 lagen en bewezen 384 unieke geldige indices per expert. Definitieve
pass vereist op zowel validation als test relatieve CE `<=2,0%` en top-1
`>=90%`.

Een pass bewijst de kwaliteit van de exact compacteerbare Q5-semantiek en opent
een fysieke bank/kernel. Hij bewijst nog geen bytes op schijf of versnelling.
