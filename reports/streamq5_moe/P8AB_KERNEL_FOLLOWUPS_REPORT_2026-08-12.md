# P8A/P8B kernelvervolg — eindbesluit

Datum: 2026-08-12

## P8A Projection-Adaptive ERVF

De selectie was bitexact over 502.144 Q8- en 1.376.256 Q5-uitgangen. Q5 koos
breedte 8 en haalde geïsoleerd een p50-ratio van `0,940184` en p95-ratio van
`0,897240`. Q8 koos per type verschillende breedtes, maar haalde p50-ratio
`1,089152` en p95-ratio `0,965074`. De vergrendelde gezamenlijke hypothese
faalde dus.

Omdat het Q5-deel zijn eigen geïsoleerde grens wel haalde, ging uitsluitend dat
deel als P8A2 naar de volledige decoder. Q8 bleef op ERVF-16.

## P8B Scale-Broadcast ERVF

De shufflevariant was bitexact over 1.376.256 Q5-uitgangen, maar was trager:
p50-ratio `1,037457`, p95-ratio `1,000973`. De extra shuffle/instructiedruk
woog zwaarder dan de vermeden schaalloads. Hypothese gefalsificeerd.

## Bewijsgrens

Dit zijn lokale, geïsoleerde projection-plane-metingen. Alleen P8A2 bepaalt of
de Q5-deelwinst in de volledige decoder blijft bestaan.

