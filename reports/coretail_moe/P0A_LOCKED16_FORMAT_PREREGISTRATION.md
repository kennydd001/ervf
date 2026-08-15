# CORETAIL-MoE P0A — werkelijk formaat op de locked16

**Vastgelegd:** 2026-08-11, vóór het bouwen van CORETAIL-bestanden.

De officiële P0 vereist 6.144 bestaande GPTQ-experts. De repository bevat
slechts de zestien canonieke FLEQ-GPTQ-experts; de overige 6.128 ontbreken na
de E2GQ-coveragefalsificatie. RTN of opnieuw verzonnen calibratiestatistieken
mogen deze broncodes niet vervangen.

P0A test daarom uitsluitend de fysieke formaatmechaniek op alle 75.497.472
beschikbare codes en bijbehorende BF16-scales.

## Vast formaat

Core, per matrixrecord:

- 64-byte header;
- ruwe BF16-scales;
- vaste little-endian nonzero-bitmap per rij;
- uint32 byteoffset per rij plus eindoffset;
- little-endian signbits uitsluitend voor nonzeros;
- CRC32 van de payload;
- recordalignment op 4.096 bytes.

Tail, per matrixrecord:

- flags voor `q=-2`, uitsluitend onder negatieve coreposities;
- blokken van 64 rijen;
- per blok zlib-9, of raw wanneer dat niet groter is;
- 32-byte indexentry met offset, lengtes, bitcount, CRC32 en codec;
- 64-byte recordheader, record-CRC32 en 4.096-byte alignment.

De decoder moet iedere code en iedere BF16-scalebit exact reproduceren. Alle
headers, indices, checksums en alignment tellen mee. De extrapolatie naar de
full bank wordt gerapporteerd maar kan de officiële P0-gate niet passeren.

P1 blijft gesloten als niet alle 6.144 canonieke GPTQ-expertbronnen bestaan.
