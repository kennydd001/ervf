# N2AS — sparse temporal Q5 preregistratie

Vastgelegd vóór het openen van de N2AS-testpartition.

## Hypothese

Voor blokken van vier tokens kan één Q5-expertrecord één keer worden gelezen en
alleen voor de tokens worden toegepast die werkelijk naar die expert routeren.
Dit behoudt de P7B/ERVF16-reductievolgorde per token, maar vermijdt herhaalde
gewichtloads wanneer de vier routerlijsten overlappen.

## Bevroren gegevens en partitions

- Routersporen: P4D, 48 lagen en vijf domeinen.
- Validatie: tokens `[512, 768)`, blokstarts 512, 576, 640 en 704.
- Test: tokens `[768, 1024)`, blokstarts 768, 832, 896 en 960.
- De fysieke Q5-bank gebruikt de eerste 32 records per laag als timingpayload;
  route-identiteiten worden per blok bijectief naar deze slots gemapt. Dezelfde
  mapping wordt voor referentie en kandidaat gebruikt.
- Seed: `120822`.

## Correctheid

Voor de eerste blokstart van ieder domein worden alle 48 lagen vergeleken.
Gate, up en down moeten bit-voor-bit gelijk zijn aan vier afzonderlijke
`q5_*_ervf16`-aanroepen. Geen tolerantie.

## Poorten

1. Validatie opent de test alleen bij bit-exactheid en een gecombineerde
   kandidaat/referentie-p50 van hoogstens `0.90`.
2. De test slaagt bij bit-exactheid, p50-ratio hoogstens `0.85` en p95-ratio
   hoogstens `0.90`.

## Claimgrens

Dit is een fysiek Q5-kernelresultaat op echte P4D-routerpatronen. Het bewijst
geen drafteracceptatie, causale beschikbaarheid van vier toekomstige tokens,
kwaliteit, volledige decoderwinst of end-to-end tokens/s.
