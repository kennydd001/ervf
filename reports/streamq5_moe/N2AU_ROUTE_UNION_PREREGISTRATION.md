# N2AU preregistratie — echte route-unie voor Temporal ERVF

Datum: 2026-08-12. Vastgelegd na N2A-kernelpass, vóór union-output.

Gebruik zonder selectie alle vijf domeinen, 48 lagen en 1.024 verzegelde P4D-
routertokens. Voor niet-overlappende blokken S=2/4/8 meet per laag het aantal
unieke experts in de unie van S top-8-routes.

De S=4-routepoort is mean-unie `<=25,6` (minstens 20% minder records dan 32)
en p95 `<=30`. Rapporteer ook de byte-lineaire projectie van N2A's Q5-
same-eighttijd, maar behandel die alleen als pessimistische extrapolatie: een
echte sparse temporal kernel voert niet alle S tokens voor iedere unionexpert
uit. Geen end-to-end- of acceptanceclaim volgt uit deze analyse.
