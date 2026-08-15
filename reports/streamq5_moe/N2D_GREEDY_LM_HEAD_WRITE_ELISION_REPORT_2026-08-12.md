# N2D — greedy LM-head write-elision sluitingsrapport

## Uitkomst

**Exact, maar fysiek trager; kandidaat gesloten en testpartition verzegeld.**
De fused per-block argmax schrijft 87,499% minder globale outputbytes, maar
verhoogt de p50-headtijd met 20,79% tegenover het huidige pad en met 21,74%
tegenover de argmax-only full-logitcontrole.

## Correctheid

- fysieke Q8-head: 151.936 × 2.048, 316.026.880 residentiële bytes;
- zestien BF16-afgeronde random inputs plus één nulvector;
- 17/17 argmaxindices exact gelijk voor A, B, C en `numpy.argmax`;
- 17/17 geselecteerde maximumbits exact gelijk voor B, C en de full logits;
- de nulvector forceerde een 151.936-voudige tie; alle paden kozen index 0;
- alle gecontroleerde waarden waren eindig.

## Fysieke ABBA-validatie

| pad | compositie | p50 ms | p95 ms | samples |
|---|---|---:|---:|---:|
| A | ERVF16 full logits + huidige logsumexp/argmax | 2,03349 | 3,86525 | 192 |
| B | ERVF16 full logits + argmax-only | 2,01754 | 3,70507 | 96 |
| C | fused 16-row candidates + globale reductie | 2,45616 | 4,24786 | 96 |

Directe ratio's:

- B/A: p50 `0,99216`, p95 `0,95856`;
- C/A: p50 `1,20786`, p95 `1,09899`;
- C/B: p50 `1,21741`, p95 `1,14650`.

De gepaarde ABBA C/A-ratio was mediaan `1,21134` en p95 `1,27771`. Dit is
ruimschoots buiten de vooraf geregistreerde validatie-openingspoorten van 1,02
en 1,05. De onafhankelijke testseed is daarom niet geopend.

## Outputverkeer

| pad | globale outputbytes/token |
|---|---:|
| A | 607.756 |
| B | 607.752 |
| C | 75.976 |

C elimineert exact 531.780 bytes tegenover A. Die reductie vertaalt zich niet
naar tijdwinst: de 316-MB-headscan domineert, terwijl de extra blokbrede
synchronisatie, kandidaatwrites en tweede reductie meer kosten dan de vermeden
full-logitwrite/read.

## Besluit

De oorspronkelijke observatie was juist op bytes maar onjuist op performance.
Greedy write-elision met één kandidaat per zestien vocabrijen is op deze fysieke
head geen geldige optimalisatie. B laat bovendien zien dat het weglaten van
logsumexp op zichzelf slechts een kleine componentwinst geeft en de grote
headweightscan niet oplost.

De zinvolle toekomstige richting is alleen een wezenlijk andere headstructuur
die gewichtrows kan overslaan. N2B heeft certificerende rowskipping al negatief
afgesloten; deze uitslag geeft dus geen reden om de LM-headroute opnieuw te
openen zonder nieuw structureel bewijs.

## Artefacten en claimgrens

- preregistratie: `N2D_GREEDY_LM_HEAD_WRITE_ELISION_PREREGISTRATION.md`;
- compilecheck: `n2d_greedy_lm_head_compile.json`;
- evaluator: `scripts/streamq5_moe/run_n2d_greedy_lm_head_write_elision.py`;
- validatie: `n2d_greedy_lm_head_validation.json`;
- onafhankelijke audit: `n2d_greedy_lm_head_audit.json`.

Claimgrens: residentiële fysieke Q8-head en greedy argmaxcomponent. Geen CE,
sampling, logitprocessors, volledige decoderwinst of tok/s-claim.
