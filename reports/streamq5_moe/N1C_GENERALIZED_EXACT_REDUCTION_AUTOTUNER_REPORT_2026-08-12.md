# N1C — Generalized Exact Reduction Graph Autotuner

Datum: 2026-08-12  
Registry-item: N003  
Verdict: **PASS op de vooraf geregistreerde lokale Q8- en Q5-poorten**

## Kernresultaat

De uitbreiding van ERVF van `{8,16,32}` naar `{4,8,16,32,64}` heeft een
nieuwe bruikbare configuratie gevonden. Alle vijf geometrieën produceren voor
beide fysieke banken exact dezelfde uitvoerbits als de oorspronkelijke
256-thread P6B-reductie.

De op validation gekozen en vóór test bevroren grafiek was:

- Q8: `head=16, k=64, o=16, q=16, router=64, v=64`;
- Q5: `gate_up=8, down=8`.

De ongeopende, thermisch gebalanceerde AB/BA-test gaf:

| Bank | ERVF-16 p50 | Autotune p50 | ratio | ERVF-16 p95 | Autotune p95 | ratio | Poort |
|---|---:|---:|---:|---:|---:|---:|---|
| Q8 | 9,2790 ms | 7,8353 ms | **0,844416** | 10,5570 ms | 9,0662 ms | **0,858782** | PASS |
| Q5 | 7,7532 ms | 7,2447 ms | **0,934416** | 8,2032 ms | 7,5769 ms | **0,923658** | PASS |

Dat is een geïsoleerde p50-speedup van `1,1843×` voor Q8 en `1,0702×` voor Q5.
Beide halen de vooraf vastgelegde grens `ratio <= 0,97` op p50 én p95.

## Exactheidsbewijs

| Bank | Breedte | Elementen | Verschillende bits | Max. absolute fout | Eindig |
|---|---:|---:|---:|---:|---|
| Q8 | 4 | 502.144 | 0 | 0 | ja |
| Q8 | 8 | 502.144 | 0 | 0 | ja |
| Q8 | 16 | 502.144 | 0 | 0 | ja |
| Q8 | 32 | 502.144 | 0 | 0 | ja |
| Q8 | 64 | 502.144 | 0 | 0 | ja |
| Q5 | 4 | 1.376.256 | 0 | 0 | ja |
| Q5 | 8 | 1.376.256 | 0 | 0 | ja |
| Q5 | 16 | 1.376.256 | 0 | 0 | ja |
| Q5 | 32 | 1.376.256 | 0 | 0 | ja |
| Q5 | 64 | 1.376.256 | 0 | 0 | ja |

De twee gemengde eindgrafieken zijn bovendien afzonderlijk opnieuw vergeleken:
Q8 heeft 0 verschillen over 502.144 elementen en Q5 0 verschillen over
1.376.256 elementen.

## Validation-selectie

Onderstaande waarden zijn p50 in milliseconden; vet is de bevroren keuze.

| Projectie | w=4 | w=8 | w=16 | w=32 | w=64 |
|---|---:|---:|---:|---:|---:|
| Q8 head | 3,2639 | 2,0154 | **1,5215** | 1,9555 | 1,7687 |
| Q8 k | 1,8020 | 1,4450 | 0,9124 | 0,7528 | **0,5739** |
| Q8 o | 6,3853 | 3,7801 | **2,2796** | 2,8125 | 2,3638 |
| Q8 q | 5,6639 | 4,1052 | **2,4755** | 3,2185 | 2,6136 |
| Q8 router | 1,6785 | 1,1900 | 0,6207 | 0,5841 | **0,4508** |
| Q8 v | 1,7570 | 1,4024 | 0,8237 | 0,7278 | **0,5522** |
| Q5 gate/up | 6,2801 | **4,4409** | 4,5516 | 7,1600 | 7,3182 |
| Q5 down | 2,5242 | **2,4848** | 2,8600 | 4,5959 | 4,7848 |

De uitkomst volgt de matrixgeometrie. De grotere Q8-projecties houden breedte
16; de smallere K/V/routerprojecties winnen met de nieuwe two-warp-breedte 64.
Q5 houdt genoeg row-parallelisme nodig om breedte 8 te verkiezen.

## Resourceverklaring

De compiler rapporteert voor Q8 respectievelijk `96/56/48/40/40` registers per
thread voor breedtes `4/8/16/32/64`. Breedte 4 betaalt dus zichtbaar voor zijn
64 virtuele accumulatoren per lane. Breedte 64 gebruikt 40 registers en 1.024
bytes statisch shared memory per block voor de exacte stride-32 cross-warpstap.
Dat verklaart waarom 64 aantrekkelijk is bij beperkte rij-aantallen, maar niet
universeel wint.

## Wat dit wel en niet bewijst

N1C bevestigt fysiek dat een vaste 256-thread FP32-reductie-DAG ook over 4 en 64
fysieke lanes bitexact kan worden gepland, en dat automatische selectie een
snellere gemengde Q8- én Q5-projectiegrafiek vindt. Dit is sterker dan P8A,
omdat breedte 64 daar niet in de zoekruimte zat en nu de drie kleinere
Q8-projectietypen wint.

Dit is nog geen volledige decoder- of tok/s-winst. De lokale geometrische
speedup over de twee banken is circa `1,126×`; daarmee is de bredere voorgestelde
publicatiepoort van `>=1,2×` over meerdere kernelfamilies/modellen/GPU's niet
bewezen. Ook INT4, BF16, RMSNorm, softmax, attention, een tweede model en een
tweede GPU-architectuur vallen buiten N1C.

De beslissende vervolgstap is een afzonderlijk vooraf geregistreerde
end-to-end-replicatie met de bevroren Q8-64/16- en Q5-8-grafiek. P8A2 testte
alleen Q5-8 terwijl Q8 ongewijzigd bleef; N1C levert nu voor het eerst genoeg
gecombineerde lokale winst om die integratie opnieuw zinvol te maken.

## Artefacten

- Preregistratie:
  `reports/streamq5_moe/N1C_GENERALIZED_EXACT_REDUCTION_AUTOTUNER_PREREGISTRATION.md`
- Uitvoer:
  `reports/streamq5_moe/n1c_generalized_exact_reduction_autotuner.json`
- Reproduceerscript:
  `scripts/streamq5_moe/run_n1c_generalized_exact_reduction_autotuner.py`
- Preregistratie-SHA256:
  `8108ca3985d098a2dd9f370e66a5eff99507e64f2015404fe8cd7a14f13235c5`
- Script-SHA256 in de uitvoer:
  `fe15d817ee68839986a9b39cbffcb3538ae998ffb35a2c98e6b42a4b68dad5f4`
