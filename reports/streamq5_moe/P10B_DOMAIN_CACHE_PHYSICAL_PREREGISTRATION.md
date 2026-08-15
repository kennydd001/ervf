# P10B preregistratie — fysieke automatische domeincache

Datum: 2026-08-12. Status: P10-simulatie geopend, fysieke P10B-output ongeopend.

## Hypothese

De in P10 causale selector (`global-12 + profile-8 + LRU`) behoudt zijn winst
wanneer iedere cachemiss en ieder profielverschil als echte 3.035.136-byte
pinned-host-naar-GPU-kopie wordt uitgevoerd.

## Test

Gebruik de ongeopende P2B-testhelft en de vastgelegde 512-token switchreeks met
vier domeinen van elk 128 tokens. Vergelijk universal-20, oracle en automatic.
De gedeelde initiële global/static preload wordt buiten de tokenstopwatch
gehouden; alle acht initiële profielexperts van oracle/automatic tellen wel op
token 0.

## Gates

- Het fysieke kopieaantal is exact gelijk aan P10-simulatie.
- Geen integriteitsfout in gekopieerde records.
- Automatic event-mean en event-p95 zijn hoogstens 10% boven oracle en minstens
  5% onder universal.

