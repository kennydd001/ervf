# P11C preregistratie — directe NVMe-tier voor Q5-experts

Datum: 2026-08-12. Status bij vastlegging: geen P11C-output geopend.

## Hypothese

De ds4-achtige NVMe-tier kan capaciteit toevoegen zonder de lokale P7-latentie-
gates te breken wanneer een volledig expertrecord van 3.035.136 bytes direct,
ongecached en 4-KiB-uitgelijnd van C: wordt gelezen en daarna via de gemeten
PCIe-route wordt gekopieerd.

## Protocol

- Windows `CreateFileW` met `FILE_FLAG_NO_BUFFERING`.
- 384 random expertreads, acht per laag, vaste seed; 16 warmups uitgesloten.
- Eén volledige sequentiële pass over alle 6.144 expertrecords / 17,367 GiB.
- Controleer samples tegen buffered bronbytes.
- Projecteer per werkelijk P7C-testtoken `wall_ms + misses × direct_read_ms`;
  de bestaande RAM-H2D-tijd zit al in `wall_ms` en wordt niet dubbel geteld.

## Gates

- nul korte reads/integriteitsfouten;
- random p95-read `<= 2 ms`;
- geprojecteerde test mean `<= 100 ms`, p95 `<= 150 ms` en throughput
  `>= 10 tok/s`.

Een pass is uitsluitend een capaciteitsbewijs, geen versnelling tegenover RAM.

