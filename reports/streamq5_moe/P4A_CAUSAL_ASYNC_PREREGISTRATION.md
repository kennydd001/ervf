# STREAMQ5-MoE P4A — causale async-H2D preregistratie

Datum: 2026-08-12. Status bij vastlegging: geen P4A-output geopend.

## Vraag

Kan de P3A-transferterm fysiek worden verborgen zonder toekomstige routes te
lezen, uitsluitend door cachemiss-kopieën van de huidige laag op een tweede
CUDA-stream te overlappen met de berekening van de cachehits van diezelfde
laag?

Dit test de uitvoerbare kern van P4B uit
`info/DATAPLANE_ONTLEDING_2026-08-12.md`. Het is geen simulatie en krijgt geen
overlapcredit: de gerapporteerde tijd is host-wandtijd tot beide streams klaar
zijn.

## Vaste input en splits

- Exact de geverifieerde P1D-bank van 17,3671875 GiB wordt volledig in pinned
  hostgeheugen geladen en gehasht.
- Exact de P3A-routes en P2C-policy: 20 statische slots per laag, plus 15
  dynamische slots in lagen 0–7 en 14 in lagen 8–47.
- Calibration blijft gesloten; validation is token 512:768 en test is token
  768:1024, voor elk van de vijf bestaande domeinen.
- De test wordt één keer geopend, uitsluitend na een volledige validation-pass.
- De cache (4.977.623.040 bytes), trunkreservering (1.541.093.376 bytes) en
  KV-reservering (402.653.184 bytes) zijn tegelijk aanwezig; minimaal 384 MiB
  scratch moet vrij blijven.
- Dezelfde deterministische FP32 beginstate wordt per token gebruikt in serial
  en async.

## Twee fysiek gemeten paden

`serial` herhaalt P3A als gelijktijdige controle: iedere misskopie en alle vier
kernels staan op één stream.

`causal_async` gebruikt twee streams, maar leest nooit route-ID’s van laag
`l+1` voordat laag `l` volledig is berekend:

1. bepaal de hits en misses van de huidige laag en voer exact dezelfde LRU uit;
2. enqueue de volledige 3.035.136-byte records van misses op de kopiestream;
3. bereken de hitexperts op de computestream met variable-N gate/up, SwiGLU en
   down-kernels, waarbij iedere uitvoer teruggaat naar zijn oorspronkelijke
   top-8-positie;
4. wacht op één CUDA-event van de kopiestream;
5. bereken de missexperts en reduceer de acht posities in dezelfde volgorde.

De opsplitsing voegt launches toe en kan dus falen; dat is onderdeel van de
test. Een één-laag-lookahead of offline-trace-prefetch is expliciet verboden.

## Vastgelegde voorspelling en poorten

De externe ontleding voorspelt 19,06 ms mean en 20,68 ms p95. Afwijking van de
mean met meer dan 15% (`[16,201; 21,919]` ms) falsificeert dat specifieke
tweetermen-overlapmodel.

Validation en test moeten elk alle primaire poorten halen:

1. de serial-missreeks is exact gelijk aan de onafhankelijk gereconstrueerde
   P3A-policy en de async-missreeks is exact gelijk aan serial;
2. het aantal fysieke H2D-records en de byte-aritmetiek zijn exact;
3. serial en async eindstates zijn bit-identiek; bij een onverwacht verschil
   gelden aanvullend `max_abs <= 2e-5` en `relative_l2 <= 1e-6` als uiterste
   correctheidsgrens, maar bit-identiek blijft apart zichtbaar;
4. alle outputs en timings zijn eindig;
5. aggregate async mean `<= 20,0 ms/token` en p95 `<= 25,0 ms/token`;
6. ieder domein heeft mean `<= 22,0` en p95 `<= 30,0 ms/token`;
7. de gelijktijdige serial-controlmean ligt binnen 15% van de overeenkomstige
   bewaarde P3A-mean;
8. co-residency en scratchgrens zijn werkelijk gehaald.

## Claimgrens

Een pass bewijst alleen causale same-layer overlap voor de fysieke expertplane
met trace-replayed routes en ongewogen top-8-reductie. Een fail sluit deze
implementatiehefboom en de 19,06-ms-projectie, maar niet alle denkbare
prefetchers. Attention, echte routergewichten, trunkcompute, KV-mutatie,
embedding/head/sampling, autoregressieve routefeedback en full-model tok/s
blijven buiten de claim.
