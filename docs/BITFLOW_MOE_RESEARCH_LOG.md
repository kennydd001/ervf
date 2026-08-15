# BITFLOW-MoE onderzoekslog

## 2026-08-11 — maximale lineaire screen vooraf geregistreerd

P0 start uitsluitend met de dominante C1/Q4-kandidaat op sequentieel opnieuw
gegenereerde studentstates. De train-, validation- en testtokenranges,
ridge-grid, BF16-repairsemantiek, resourcegates en 50%-hard-stop zijn gelockt.
Geen C0-, route-FiLM-, Q3- of syndromewerk vóór C1 beide progression gates haalt.

## 2026-08-11 — verifierpoging 001 ongeldig

De eerste onafhankelijke ruwe-logitcontrole gebruikte een absolute
CPU-versus-CUDA CE-tolerantie van 1e-9. BF16-logits gaven door een andere
reductievolgorde circa 1,9e-5 verschil. De poging blijft append-only bewaard;
alleen de numerieke vergelijking is vóór herhaling op 5e-5 gezet. Uitkomsten,
logits en gates zijn niet gewijzigd.

Verifierpoging 002 slaagde 22/23. Alleen de afgeleide recoveryratio week
7,7e-5 af doordat CE-ruis door de kleine Q4-schade wordt gedeeld. Daarom krijgt
uitsluitend die afgeleide ratio een tolerantie van 1e-3; ruwe metrics behouden
5e-5. Ook poging 002 blijft bewaard.

## 2026-08-11 — P0 lineaire tak gesloten

C1/Q4 miste de 50%-progression gate overtuigend: validation-recovery was
-645,95% en test-recovery -395,02%. De test-top-1-overeenkomst was 75,00% en
de late-laagexplosieratio 6,4635. De historische Q4-testbaseline werd exact
gereproduceerd. De definitieve onafhankelijke controle slaagde 23/23.

Conform preregistratie zijn C0, C2, Q3, syndrome en P1 niet geopend. Dit
falsifieert de geregistreerde data-gelimiteerde dense lineaire C1-route, niet
iedere denkbare niet-lineaire of veel-data BITFLOW-variant.
