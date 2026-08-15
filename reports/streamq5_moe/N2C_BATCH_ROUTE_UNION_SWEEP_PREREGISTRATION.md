# N2C — fysieke batch/route-union-sweep

Datum: 2026-08-12. Vastgelegd vóór N2C correctness- of timinguitvoer.

## Hypothese

Een batch van meerdere reeds beschikbare tokenactivaties kan Q8-trunkgewichten
en overlappende Q5-expertgewichten over tokens hergebruiken. Op echte
P4D-routersporen kan daardoor de gecombineerde residentiële Q8+Q5-componenttijd
per batch dalen, ook als route-uniegroei een deel van de Q5-winst opeet.

## Kandidaten en fysieke payloads

- Sweep: `S = 2, 4, 8, 16` tokens.
- Referentie: S afzonderlijke Q8 ERVF-16-planes en S afzonderlijke Q5
  `aligned32x2`-planes uit N1B.
- Kandidaat: één temporele Q8-plane en per laag één sparse-temporal Q5-plane.
- Q5 decodeert dezelfde fysieke Q5-codes en BF16-schalen. Echte router-ID's
  worden per blok bijectief op een vaste fysieke payload van 80 expertrecords
  gemapt; dezelfde mapping geldt voor referentie en kandidaat. De geselecteerde
  partitions hebben maximaal 76 unieke experts.
- Q8 gebruikt de volledige fysieke P6A-devicebank.
- Activaties: vaste seed `120823`, FP32, maximaal `[16, 4096]`.

De sparse kernel leest ieder fysiek Q5-pack per union-expert eenmaal en voert
MAC's alleen uit voor de tokens die werkelijk naar die expert routeren. N1B's
aligned-32-bit packreconstructie wordt in beide Q5-armen gebruikt.

## Routerpartitions

- P4D-sporen, alle 48 lagen en vijf domeinen.
- Validatieblokstarts: 512, 576, 640 en 704.
- Testblokstarts: 768, 832, 896 en 960.
- Per start is één S-breed aaneengesloten tokenblok gebruikt.
- Alle 48 routebestanden moeten met de P4D-capturehashes overeenkomen.

## Correctheid

Vóór timing worden voor ieder S gecontroleerd:

- Q8: iedere fysieke projectie volledig;
- Q5: eerste blokstart van ieder domein, alle 48 lagen;
- gate, up, down en Q8 moeten bit-voor-bit gelijk zijn aan de sequentiële
  referentie; geen tolerantie en uitsluitend eindige outputs.

## Gesloten meetpoorten

- Validation: twee warmups en twaalf afwisselend geordende AB/BA-paren.
- Per S worden Q5-only en gecombineerd Q8+Q5 apart gemeten.
- Een S opent zijn test bij volledige bitexactheid en gecombineerde
  validation-p50-ratio `<= 0.98`.
- Test: drie warmups en dertig afwisselend geordende AB/BA-paren.
- Een S slaagt bij gecombineerde test-p50-ratio `<= 0.95`, p95-ratio
  `<= 1.00` en blijvende bitexactheid.
- De sweep slaagt als minstens één vooraf vastgelegde S slaagt. De uiteindelijke
  winnaar is onder geslaagde S de laagste gecombineerde kandidaattijd per token
  op test; zonder testpass wordt geen winnaar uitgeroepen.

## Claimgrens

Dit is een fysieke residentiële componenttest met echte routerpatronen. Een
pass bewijst geen causale beschikbaarheid van toekomstige tokens,
speculatieve-decodeacceptatie, batchkwaliteit, volledige decodersnelheid,
host/device-overlap of externe SOTA.
