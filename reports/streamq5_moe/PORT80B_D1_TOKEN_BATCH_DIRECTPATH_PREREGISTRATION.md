# PORT80B-D1 — token-batch direct-path decomposition preregistration

Datum: 2026-08-12. Dit is de eigen hypothese van de hoofdonderzoeker,
vastgelegd voordat `nemotron.txt`, `RAND_VAN_WAT`, het aangeleverde DIRECTPATH-
pack, `gpt.txt` of het SplitTree-rapport inhoudelijk worden gebruikt en voordat
een D1-timingoutput bestaat.

## Hypothese

P0 stuurde voor een zero-cachetoken 480 afzonderlijke expertrecords van
2.027.520 bytes via acht roterende pinned windows. De gemeten p95 van 73,544 ms
kan daarom deels uit transferdispatch en windowrendezvous komen. Wanneer exact
dezelfde 973.209.600 bytes eerst in één groot pinned tokenbuffer staan, kan één
aaneengesloten H2D-copy de fysieke 45-ms-poort alsnog halen.

## Bevroren fysieke input

- de bestaande niet-sparse P0-bank van exact 49.925.652.480 bytes en SHA256
  `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`;
- dezelfde SplitMix64-routefunctie en expertvolgorde als P0;
- token 10.000 levert 48 lagen × 10 unieke routed experts;
- iedere arm gebruikt exact dezelfde geordende 973.209.600 bronbytes en een
  bestemming met dezelfde lengte;
- één pinned bronbuffer en twee devicebuffers; allocaties, routeplanning en
  mmap-gather staan buiten de H2D-events.

## H2D-armen

1. `record480`: 480 asynchrone copies van 2.027.520 bytes, één synchronisatie
   na de volledige token;
2. `layer48`: 48 asynchrone copies van 20.275.200 bytes;
3. `token1`: één asynchrone copy van 973.209.600 bytes.

Voor iedere arm zijn 10 warmups toegestaan. Daarna volgen 120 CUDA-event-
metingen in per ronde geroteerde volgorde, om de ronde omgekeerd. De volledige
devicebuffer van iedere arm moet vóór timing bytegelijk zijn aan de `token1`-
referentie.

## Mmap→pinned stagingdiagnose

Dezelfde pinned tokenbuffer wordt voor 32 deterministische tokens gevuld met
480 `np.copyto`-operaties uit de read-only mmap. Eén volledige, ongetimede
ronde over dezelfde tokens warmt precies die bronpagina's; de tweede ronde is
de page-resident timingreeks. Staging en H2D worden niet tegelijk uitgevoerd.

`max(staging_p95, token1_H2D_p95)` is alleen een ideale perfecte-overlapgrens;
de som is de volledig seriële grens. Er wordt geen werkelijk double-buffer-
overlapresultaat geclaimd.

## Poorten

D1 slaagt als directe H2D-component alleen wanneer:

- alle drie volledige devicebuffers bytegelijk zijn;
- elke arm exact 120 eindige samples heeft;
- `token1` p95 `<=45 ms`;
- `token1_p50 / record480_p50 <=0,80` en
  `token1_p95 / record480_p95 <=0,90`.

De bredere direct-path feasibility-poort vereist daarnaast de ideale
overlapgrens `max(staging_p95, token1_p95) <=45 ms`. Een H2D-pass met een
stagingfail bewijst batching, maar nog geen uitvoerbaar tokenpad. D1 bevat geen
expertcompute, echte router, modelkwaliteit, dense shell of end-to-end tok/s.
