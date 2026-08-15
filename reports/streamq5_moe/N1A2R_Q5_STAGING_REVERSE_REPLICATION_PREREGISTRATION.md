# N1A2R preregistratie — kandidaat-eerst replicatie

Datum: 2026-08-12. Vastgelegd na N1A2-pass, vóór N1A2R-output.

N1A2 draaide baseline vóór kandidaat. N1A2R herhaalt exact dezelfde 7+256
inputtokens in een nieuw proces, maar voert de staged kandidaat eerst en de
referentie daarna uit. Dit test order-, thermische en host-file-cachebias.

Gates: voorspellingen, misses en KV-digest exact; kandidaat mean en p95 elk
`<=0,95×` baseline. Alleen wanneer N1A2 én N1A2R slagen, opent de 10K-run.
