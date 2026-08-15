# Offload-roofline onderzoekslog

## 2026-08-11 — claims opgesplitst en meetprotocol gelockt

P-B wordt als exploratieve heranalyse op reeds geopende HERA-routes uitgevoerd.
P-C meet eerst uitsluitend de nieuwe lokale pinned-H2D-leg. P-A, de volledige
P-C-decode, P-D en P-E krijgen een expliciete inputaudit; ontbrekende modellen,
packed formats, runtimes of route-identiteiten worden niet door projecties
vervangen.

## 2026-08-11 — P-C-verifierpoging 001 stopte vóór output

De eerste verifierpoging berekende alle controles maar stopte vóór het schrijven
van een artifact: een `numpy.bool_` was niet JSON-serialiseerbaar. De ruwe
hardwaremeting is niet gewijzigd. Alleen de verifier cast voortaan elke
controle expliciet naar een Python-boolean.

## 2026-08-11 — P-E geopend met afgescheiden calibratie

De laag-26-componenttrace en echte BF16-gewichten maken een directe P-E-test
mogelijk. De expert-specifieke spectrale permutatie wordt alleen op validation
indices 256–1023 geleerd. Validation/test 0–255 blijven evaluatie. De oude
tile-64, globale neuronoracle en nieuwe gepermuteerde tile-64 worden met gelijk
budget en dezelfde exacte bijdragescore vergeleken.

## 2026-08-11 — P-E-poging 001 ongeldig door batchvorm

Poging 001 berekende calibratie en evaluatie samen in één 1.280-tokenbatch.
Hoewel alle inputhashes gelijk waren, veranderde die BF16-GEMM-batchvorm de
historische tile-KL met 8,67e-5 op validation en 2,55e-4 op test; daardoor
faalde de vooraf vastgelegde reproductiecontrole. Resultaat, rapport en
permutaties blijven als `attempt_001` bewaard. De herhaling berekent de
768-tokencalibratie en oorspronkelijke 512-tokenevaluatie in aparte calls.

Poging 002 reproduceerde vervolgens de neuron- en tilemaskers exact, maar de
historische KL bleef door BF16-kerneluitvoering 7,05e-6 en 6,67e-5 afwijken.
Deze poging blijft eveneens bewaard. Poging 003 herstelt daarom ook exact de
oorspronkelijke 43-policyvolgorde en policybatching; gates blijven ongewijzigd.

## 2026-08-11 — P-B, P-C, P-D en P-E afgesloten

P-B faalde: alle vijf gemiddelden waren ≤3, maar math/instruction hadden p99
18/31 en slechts 3/20 wissels herstelden binnen 200 tokens. Onafhankelijke
replay: 14/14.

P-C mat op de lokale PCIe 5.0 ×8-link 26,1589 GB/s mediaan bij 512 MiB. Onder
de externe 27,28-GB-trunkaanname is het plafond 0,9589 tok/s; de volledige claim
blijft geblokkeerd zonder K3-trunkmeting en decode. Verificatie: 15/15.

P-D’s Qwen-uniecurve is gemeten, maar acceptatie niet. De aangeleverde uniforme
K3-berekening is gecorrigeerd van 118,6 naar 120,279 experts. De artifactaudit
slaagde 15/15; P-A, volledige P-C en P-D blijven door ontbrekende runtimes of
checkpoints geblokkeerd.

P-E reproduceerde de historische baseline exact. De spectrale tile-64-ratio
werd 5,014× op validation en 8,322× op test, ruim boven 1,20×. De BF16-
permutatiereconstructie was bovendien niet bit-exact. Verificatie: 18/18.

Alle 153 regressietests slagen. De campagne bevat nuttige mechanische en
roofline-evidence, maar geen bewezen Eureka.
