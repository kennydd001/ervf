# P12R2 preregistratie — Pinned-Window Streaming onder 32 GiB

Datum: 2026-08-12. Status: P12 en P12R allocatiefouten geopend; geen P12R2-
output geopend.

## Hypothese

De volledige 17,367-GiB expertbank hoeft niet tegelijk private, page-locked
commit te zijn. Map de 48 Q5-laagfiles read-only en gebruik acht pinned vensters
van elk exact één expertrecord (8 × 3.035.136 bytes). Voor iedere miss wordt het
ongewijzigde record naar een vrij venster gekopieerd en daarna async naar zijn
GPU-slot. Een venster wordt pas na zijn eigen CUDA-event hergebruikt.

Samen met P12R Q8-staging moet dit de P7-semantiek onder 32 GiB brengen zonder
de top-8-overlap te verwijderen.

## Gates

Alle P12-gates blijven gelden. Daarnaast:

- alle 48 mapped laaghashes zijn exact gelijk aan P1D;
- pinned expertvensters zijn exact 24.281.088 bytes;
- blijvend pinned Q8-hostgeheugen is 316.026.880 bytes;
- staging wijzigt geen prediction-, miss- of 4K-KV-integriteit.

Een pass bewijst capaciteit en duur op deze machine. Snelheid wordt volledig in
de 10.000-token-stopwatch meegemeten.

