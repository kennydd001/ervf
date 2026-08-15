# P12R preregistratie — Residency-Lifetime Staging onder 32 GiB

Datum: 2026-08-12. Status: P12 allocatiefout geopend; geen P12R-output geopend.

## Nieuwe hypothese

P12 faalde omdat de volledige 1.565-MB Q8-bronbank pinned bleef terwijl de
definitieve 1.249-MB GPU-trunk werd gealloceerd. Voor device-resident records is
die hostlevensduur niet nodig. P12R houdt alleen de 316.026.880-byte embedding
pinned en kopieert iedere Q8-devicefile via één herbruikbare pinned stagingbuffer
van maximaal 8.519.680 bytes naar de definitieve trunk.

## Exactheid

Recordhashes en geaggregeerde Q8-hash moeten gelijk blijven. Codes, schalen,
offsets in de GPU-trunk, kernels, cache, KV, router en decoder zijn identiek aan
P7C. De stagingbuffer wordt pas hergebruikt na synchronisatie.

## Gates

Alle P12-gates blijven ongewijzigd. Daarnaast:

- blijvend pinned Q8-hostgeheugen is exact 316.026.880 bytes;
- stagingbuffer is maximaal 8.519.680 bytes;
- P7C-aggregaat-Q8-hash blijft gelijk.

Dit is een geheugenlevensduurtechniek, geen quantisatie- of kwaliteitswijziging.

