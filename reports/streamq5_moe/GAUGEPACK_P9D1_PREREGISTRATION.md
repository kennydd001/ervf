# GaugePack P9D-1 — codec/oracle-preregistratie

Datum: 2026-08-12. Status bij vastlegging: GaugePack-output bestaat nog niet.

## Vraag en vaste scope

Kan de reeds geslaagde P9B-semantiek fysiek compact worden gecodeerd zonder de
Q5-code, BF16-schaal, oorspronkelijke quantisatiegroep of nulsemantiek van een
survivor te wijzigen?

De eerste vaste proef gebruikt laag 0, experts 0 tot en met 15. De bron is het
lokale BF16-checkpoint, het immutable P9B-keepmasker en de onafhankelijk
geverifieerde P1D-Q5-bank. Er wordt geen full-depth-, kwaliteits-, kernel- of
wall-clockclaim gemaakt.

## Referentiesemantiek

- P9B zet alle niet-geselecteerde gate/up-rijen en down-kolommen op nul en
  voert daarna de bestaande group-128 Q5-quantisatie uit.
- Q5-codes worden tegen de FP32-maxabs-schaal gekozen; de persistente schaal
  wordt daarna naar BF16 afgerond en die BF16-schaal bepaalt de decode.
- Gate/up-survivors behouden daardoor exact hun P1D-code en -schaal.
- Bij down blijft iedere positie in haar oorspronkelijke groep van 128. Omdat
  P9B vóór quantisatie nulde, kan de groepsmaximumwaarde veranderen. Daarom
  wordt de P9B-downreferentie opnieuw uit brongewicht plus keepmasker berekend;
  P1D-downbytes worden niet ten onrechte als P9B-downbytes aangemerkt.

## Codec

- Eén deterministisch binair bestand met een vaste globale header.
- Per expert: het gesorteerde `uint16`-keepmasker, zes survivor-aantallen per
  oorspronkelijke groep en CRC's.
- Gate/up: alleen survivorrijen, met hun letterlijke Q5-codes en BF16-schalen.
- Down: alleen survivorcodes in keepvolgorde, maar alle zes oorspronkelijke
  BF16-groepsschalen per outputrij.
- Decoder reconstrueert de logische `[768,2048]`, `[768,2048]` en
  `[2048,768]` P9B-matrices; ontbrekende waarden zijn exact nul.

## Vooraf vastgelegde poorten

De proef slaagt uitsluitend als alle onderstaande controles slagen:

1. alle 16 maskers hebben exact 384 unieke indices in `[0,768)`;
2. maskerindices en afgeleide group-ID's overleven encode/decode exact;
3. nul survivorcode-mismatches en nul raw-BF16-scalebitmismatches tegenover
   een afzonderlijk berekende P9B-referentie;
4. nul BF16-bitmismatches over alle elementen van de drie volledig
   gereconstrueerde dichte P9B-matrices;
5. gate/up-survivorcodes en -schalen matchen letterlijk met P1D;
6. alle headers, payloadgroottes en CRC32-controles zijn geldig;
7. SHA-256 van het na schrijven opnieuw gelezen bestand is vastgelegd;
8. fysieke byteratio tegenover dezelfde 16 volledige P1D-experts is `<=0,51`.

Iedere mismatch sluit P9D-1. Een pass bewijst uitsluitend dat de representatie-
bridge voor deze 16 experts bestaat. Snelheid, full-depth equivalentie en de
volledige 6.144-expertbank blijven dan afzonderlijke vervolgproeven.
