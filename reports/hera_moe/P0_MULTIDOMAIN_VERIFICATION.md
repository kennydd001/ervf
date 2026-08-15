# HERA-MoE P0 — onafhankelijke verificatie

Uitkomst: **PASS (15/15)**.

De vijf vooraf gelockte domeinen maken **6,081/6.144** experts hot. De geprojecteerde resident weights zijn **7.167 GiB**, oftewel **1.417 GiB boven** de 5,75-GiB-gate.

Alle 48 artifact- en rapporthashes, 240 routedatasets, counts, invocationtotalen, uniongroei, cold-callpercentielen en geheugenformules zijn onafhankelijk herberekend. General reproduceert E2GQ exact.

Daarmee is uitsluitend de statische multidomain `count>=128`-union gefalsificeerd vóór kwaliteitstuning. Een dynamische of domeingeconditioneerde cachearchitectuur is niet getest en mag niet als gered resultaat worden gepresenteerd.
