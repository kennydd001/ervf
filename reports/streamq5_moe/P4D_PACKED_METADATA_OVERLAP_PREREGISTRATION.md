# STREAMQ5-MoE P4D — packed-metadata full-overlap preregistratie

Datum: 2026-08-12. Status bij vastlegging: P4D-inputs en routes nog niet
gegenereerd; geen P4D-timingoutput geopend.

## Definitieve implementatiehypothese

De succesvolle P4A-architectuur blijft rekenkundig ongewijzigd: hits doorlopen
parallel gate/up, SwiGLU en down terwijl missrecords op een tweede stream
kopiëren; daarna doorlopen misses dezelfde drie kernels en volgt één reductie.

Alleen CPU/metadata-overhead verandert:

- laagbases worden eenmaal vooraf berekend;
- full slots, hit slots/posities en miss slots/posities staan in één pinned
  record van 40 int32-waarden;
- per laag is er één 160-byte metadata-H2D in plaats van vier `.set`-calls.

Geen projectie wordt geserialiseerd en het volledige hitcompute-overlapvenster
blijft behouden. De hypothese is mean <= 20,0 ms en p95 <= 25,0 ms, bit-exact.

## Nieuwe data vóór timing

Een nieuwe Q5+INT8-routecapture wordt gemaakt op vijf nieuwe reeksen van 1.024
tokens. Elk 128-tokencontext moet exact disjoint zijn van alle eerder gebruikte
STREAMQ5-route-inputs. Bronoffsets worden vóór capture gelockt. Per laag worden
alle vijf routedomeinen opgeslagen en gehasht.

- calibration: 0:512, uitsluitend voor statische top-20;
- validation: 512:768;
- eenmalige test: 768:1024, alleen na volledige validation-pass.

## Fysieke constanten

Exact P1D-bank, P2C 20+15/14 LRU, 3.035.136 bytes per missrecord,
4.977.623.040-byte cache, 1.541.093.376-byte trunkreservering,
402.653.184-byte KV-reservering en minimaal 384 MiB scratch. Serial en async
krijgen dezelfde deterministische FP32-states. Geen toekomstige route wordt
gelezen.

## Primaire poorten

1. verse routecapture volledig, eindig, alle IDs geldig en inputcontexten
   disjoint;
2. serial en async exact dezelfde missreeks;
3. H2D-records en bytes exact;
4. eindstates bit-identiek; daarnaast `max_abs <= 2e-5` en
   `relative_l2 <= 1e-6`;
5. alle outputs/timings eindig;
6. async aggregate mean `<= 20,0`, p95 `<= 25,0 ms`;
7. elk domein mean `<= 22,0`, p95 `<= 30,0 ms`;
8. serial mean `<= 35 ms`, positieve misshelling;
9. async mean binnen 15% van 19,06 ms;
10. co-residency en minimaal 384 MiB scratch werkelijk gemeten.

## Claimgrens

Een pass repliceert causale full-overlap op volledig nieuwe routes en bewijst
de fysieke routed-expert dataplane onder 20 ms voor deze implementatie. Het is
nog geen volledig model: echte routergewichten in de loop, attention/trunk,
KV-mutatie, embedding/head/sampling en autoregressieve routefeedback ontbreken.
