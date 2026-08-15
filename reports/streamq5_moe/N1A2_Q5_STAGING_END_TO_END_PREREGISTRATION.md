# N1A2 preregistratie — Q5 activation staging, volledige decoder

Datum: 2026-08-12. N1A-Q5 test is geopend en positief; N1A2-output ongeopend.

## Wijziging

Start van de bewezen P13C-runtime. Alleen Q5 gate/up- en down-kernels krijgen
per block dezelfde shared-activation staging als N1A. Q8, attention EVT-PM,
pinned-window streaming, router, cachepolicy, banken, BF16-grenzen en alle
overige kernels blijven ongewijzigd.

## Werkbelasting en referentie

- activeer `general`;
- voer de zeven vergrendelde P7-rolloutprompttokens uit;
- voer daarna de eerste 256 vergrendelde P7-feedbacktokens uit, dus zonder
  sampling- of route-oracle;
- gebruik contextposities 0..262 en meet alleen de laatste 256 calls;
- vergelijk token voor token met de bestaande P13C-runtime op exact dezelfde
  invoerreeks, ieder vanuit een schone runtime/cache.

## Gates

- 256/256 voorspellingen, missenaantallen en finale KV-digest exact gelijk;
- kandidaat mean en p95 elk `<=0,95×` referentie;
- geen niet-eindige token en geen runtimefout.

Een fail sluit alleen deze shared-staging-integratie. Een pass bewijst nog geen
10K thermische verbetering; daarvoor is eerst een korte integratiepass nodig.
