# N3B — VRAM/KV/cache-Pareto preregistratie

Vastgelegd vóór analyse van de P4D-testpartition.

## Doel

Bepaal voor BF16-KV bij context 4K, 8K, 16K en 32K hoeveel fysieke
Q5-expertslots naast de P13-Q8-trunk en een vaste 384 MiB-reserve in 8 GiB VRAM
passen, en welke router-missratio daaruit volgt.

## Bevroren invoer

- `free_before_bytes` en `trunk_device_bytes` uit de fysieke P7C-runtime.
- KV-layout: 48 lagen × K/V × 4 KV-heads × context × 128 × BF16.
- Q5-record: 3.035.136 bytes.
- Routes en hashes: P4D, alle vijf domeinen en 48 lagen.
- Calibratie `[0,512)`, validatie `[512,768)`, test `[768,1024)`.

## Selectie en test

Per contextbudget worden slots zo gelijk mogelijk over lagen verdeeld. Voor
iedere mogelijke uniforme vaste-setgrootte wordt de vaste set uitsluitend op
calibratie gekozen; validatie kiest de vaste/LRU-splitsing met de laagste
missratio. Alleen die keuze wordt op test geëvalueerd.

## Poorten

- Rekenkundige allocatie moet intern exact sluiten en iedere laag moet ten
  minste acht slots hebben.
- De 8K-configuratie moet fysiek berekenbaar zijn met 384 MiB reserve.
- 32K wordt als `current_static20_compatible` gemarkeerd alleen wanneer minimaal
  20 slots per laag passen; anders blijft het een herontwerp, geen runtimeclaim.

## Claimgrens

Capaciteits- en routertracebewijs, geen fysieke 8K/32K-decode, attentiontiming,
kwaliteit of end-to-end throughput.
