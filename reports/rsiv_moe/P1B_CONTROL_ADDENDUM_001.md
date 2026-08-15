# RSIV-MoE P1B controladdendum 001

Vastgelegd op `2026-08-11T08:00:51.7306230Z`, vóór validationpoging 002 en vóór
opening van P1B-test.

P1B-preregistratie vereist de reeds geslaagde upstream P1A-operatoridentiteit
en nieuwe full-rank FP64-reconstructie van alle 1.024-tokenprefix-`x/z`-bases.
Beide slagen. De evaluator voegde daarnaast een niet-geregistreerde herhaling
van de absolute FP32-`x/g/u/z/y`-extreemgrens toe op de grotere P1B-sample.

Deze extra herhaling blijft volledig gerapporteerd, maar is vanaf poging 002
diagnostisch. Zij mag de lock niet ongeldig maken. De relatieve errors blijven
ruim onder `2e-5`; de grotere absolute maxima veranderen geen algebra, route,
rank of primary gate.

Er verandert niets aan data, contexts, rank-/thresholdgrid, byteboekhouding,
selectieregel of validation→testslot. De raw capture blijft
`1ef80fac0602c4ba146127643d243744573163a85621979df69e5b60c92a90fc` en de
ongeldige v1-lock blijft bewaard als
`reports/rsiv_moe/p1b_long_prefix_validation_selection.json`, SHA-256
`192be7621be51d3679f0e65613b5aae5539b4e75779ad5534f81083594a8de84`.

