# H4 Residual Syndrome Sketch / SketchGate

## Definitief oordeel

**H4 is gefalsificeerd op laag 26.** De trace-anchored replicatie haalt met de
validatiegekozen `r=64` Rademacher-sketch wel ruim 80% van de exacte
25%-oracle-KL-winst, maar mist ongeveer een kwart van de vooraf gedefinieerde
high-damage expertinvocaties. De vaste grens was maximaal 1%. Bovendien draagt
de downmatrix op beide replicatiesplits minder dan de vereiste 70% van het
volledige matrix-oraclesignaal en kost de gemeten eager sketchcompute in het
expliciete hardwaremodel meer dan 10% van de vermeden transfertijd.

Er volgt daarom geen uitbreiding naar lagen 13/23 of OOD en geen complexere
first-order sketch binnen H4.

## Trace-anchored replicatie

| Vaste metric | Validatie 256–511 | Test 256–511 | Gate |
|---|---:|---:|---:|
| all-Q3 KL | 0,011883 | 0,012879 | — |
| perfecte 25%-oracle-KL | 0,002355 | 0,002378 | — |
| SketchGate seed 20260810 KL | 0,003846 | 0,004241 | — |
| oracle-recovery, primaire seed | **84,35%** | **82,26%** | ≥80% |
| minimum recovery over vijf seeds | **83,74%** | **82,18%** | ≥80% |
| high-damage-FN, primaire seed | **22,73%** | **24,68%** | ≤1% |
| maximum FN over vijf seeds | **22,73%** | **24,68%** | ≤1% |
| down-only aandeel oraclewinst | **53,76%** | **65,89%** | ≥70% |

De recoverycomponent is dus reproduceerbaar positief, maar niet de
schade-eventidentificatie waarop de hypothese en gate rusten. De 154
high-damage events per split zijn de hoogste 10% positieve exacte
single-upgrade-KL-voordelen; er waren respectievelijk 946 en 1.010 positieve
events, zodat geen fallbackdefinitie nodig was.

De gekozen schedule heeft relatieve CE `+0,2630%` en top-1 `96,48%` op
validatie; op test relatieve CE `−0,0807%` en top-1 `97,27%`. Dit toont dat het
gemiddelde kwaliteitsfront bruikbaar is, maar mag de gefaalde FN-gate niet
vervangen.

## Matrixattributie en baselines

| 25%-oraclefamilie | Aandeel volledige oraclewinst validatie | Test |
|---|---:|---:|
| gate-only | 55,64% | 50,76% |
| up-only | 56,76% | 59,08% |
| down-only | 53,76% | 65,89% |
| gate+up | **79,86%** | **74,30%** |
| alle matrices | 100% | 100% |

Gate+up domineert down-only op beide splits. Dat verklaart waarom een zuivere
downresiduschets gemiddelde KL redelijk rangschikt maar de werkelijk
schadelijkste full-matrix-events niet betrouwbaar vangt.

Op hetzelfde venster haalt de niet-deploybare exacte Q4-outputenergie
83,96%/81,55% recovery met 22,73%/25,32% FN; routergewicht haalt
82,12%/82,87% met 29,87%/27,92% FN. SketchGate is dus geen kwalitatieve sprong
ten opzichte van eenvoudige of niet-deploybare controles. De historische
1.024-token ridge- en progressive-predictors haalden 86,39% en 87,88%
recovery; die andere vensters zijn alleen context en geen same-window gate.

## Metadata, hardware en controles

- geselecteerde metadata: 5.775.360 bytes, `0,08345` effectieve bit per
  origineel routed expertgewicht; **metadata-gate geslaagd**;
- int8-syndroomquantisatie: gemiddelde NRMSE `0,00804`;
- hardwaremodel: 0,0461 ms mediane sketchcompute tegenover 0,1896 ms gemeten
  transfer van 4.866.048 vermeden vierde-bitbytes; ratio `24,30%`, dus de
  `<10%`-gate **gefaald**;
- natuurlijke route-ID's en routergewichten reproduceren exact;
- opgeslagen Q3/Q4-output wordt bitexact gebruikt in de replicatie;
- officiële teachercontrol geeft KL/CE exact nul en top-1 exact één;
- iedere gekozen schedule-KL uit de 64-maskertabel is exact gelijk aan de
  afzonderlijke volledige-vocabulaire-evaluatie.

Het hardwaremodel is geen packed runtime en geen snelheidsclaim.

## Waarom twee resultaatbestanden bestaan

De eerste vooraf geregistreerde volledige run
`sketchgate.json` miste de inhoudelijke gates al, maar haar extra herberekening
van oude Q3/Q4-componenten overschreed de vóór die run vastgezette
batchvormtolerantie. Dat bestand blijft ongewijzigd bewaard en telt niet als
sluitende falsificatie.

Vervolgens is vóór inspectie van nieuwe vensters een afzonderlijke
trace-anchored replicatie geregistreerd. Zij gebruikt splitposities 256–511,
bouwt kandidaten in de originele MoE-som en sluit alle vereiste controls. Deze
replicatie draagt het definitieve H4-oordeel.

## Raw bewijs en stop/go

- beslissende JSON: `13.946.126` bytes, SHA-256
  `cae54755ad40a4dd1e46a95075f0331a82b58ff45eedaa15e98de3b1459f6f90`;
- oorspronkelijke control-falende JSON: `13.931.785` bytes, SHA-256
  `98fa42ed2987c31ed89a2c2d00f05aecbf1d8fcbac922ce16bb63613b4bbf0b9`;
- totale replicatiecompute: `13,95 s`; piek toegewezen VRAM
  `1.263.463.424` bytes;
- alle 64-masker-KL-tabellen, 50 sketchconfiguraties per split, vijf random
  schedules, gepakte masks, tokenseries, hashes, hardware en software staan in
  de JSON.

**Stop H4:** geen spread, OOD, calibrator of complexere gate/up/down-sketch.
**Go onderzoeksprogramma:** vervolg de vooraf bepaalde P0-volgorde met H2
Block-Coalescing Oracle. Het deels positieve recoveryresultaat blijft als
diagnostiek bestaan, maar is geen Eureka.
