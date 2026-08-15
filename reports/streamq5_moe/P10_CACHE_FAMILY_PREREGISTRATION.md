# P10 preregistratie — causale cachefamilie en domeinprofielen

Datum: 2026-08-12. Status bij vastlegging: geen P10-output geopend.

## Data en splits

Gebruik uitsluitend de fysieke P2B-routecapture: vijf domeinen, 48 lagen, 128
experts, top-8, 1.024 tokens per domein. Tokens 0–511 zijn validation; 512–1023
zijn test. Alle statische sets, frequenties, pair-tabellen en selectiecriteria
worden alleen op validation geleerd.

## Budget

Iedere gewone kandidaat gebruikt exact 20 statische slots per laag en in totaal
680 dynamische slots (15 voor lagen 0–7, 14 voor 8–47), dus 1.640 slots totaal.
Water-filling mag de 680 dynamische slots anders over lagen verdelen. Iedere
prefetch en profielwissel telt als een fysieke recordkopie van 3.035.136 bytes.

## Kandidaten

1. baseline domain-static + LRU;
2. LFU;
3. TinyLFU-admission;
4. 2Q;
5. causale pair-prefetch;
6. validation-water-filling over lagen.

Een kandidaat wordt op validation geselecteerd als hij mean én p95
request-misses minstens 5% verlaagt en niet meer totale records kopieert dan de
baseline. Op test gelden vooraf: minstens 3% lagere mean én p95 en geen extra
kopieën.

## Domeinconditionering

Vergelijk universal-20, oracle `global-12 + profile-8` en een causale automatische
selector op pure, gemengde en domeinswitchsequenties. De selector gebruikt
uitsluitend een rollend venster van reeds geziene routes en validation-
frequenties. Profielwissels laden de setverschillen fysiek in de boekhouding.

De automatische policy slaagt als zijn totale kopieën hoogstens 5% boven de
oracle liggen én minstens 5% onder universal-20, zowel geaggregeerd als op de
switchsequentie.

