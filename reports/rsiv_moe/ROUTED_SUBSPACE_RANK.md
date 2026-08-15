# RSIV-MoE P1 routed-subspace-rankcensus

## Uitkomst

**P1-verdict: `screen_negative_v2`.**

De vooraf vastgelegde P1-screen faalt op validation of op de eenmalig geopende test voor dezelfde kandidaat. P2 op V2 wordt niet geopend.

## Bevroren kandidaat

- Rankcap: `4`.
- Gedeelde residualthreshold: `0.001`.
- Selectietype: `diagnostic_validation_failure`.
- Capture-SHA-256: `8c532434d50df8bd65691a72351a21084bdf00d5b68406f27806379ad9a67906`.
- Selection-lock-SHA-256: `cc8125000e775effbd09e7fb70866d5595b85af7e4642b477a22583c38bc53f5`.

## Validation en eenmalig geopende test

| Evaluatie | Split | Double-fast | Koude-bytereductie |
|---|---|---:|---:|
| Offline trainbasis | validation | 0.000% | 1.000× |
| Causale 96→32-prefix | validation | 0.000% | 1.000× |
| Offline trainbasis | test | 0.000% | 1.000× |
| Causale 96→32-prefix | test | 0.000% | 1.000× |

Primaire eis voor beide evaluaties: minstens 92% double-fast en minstens 10× minder geprojecteerde koude expertbytes bij rank maximaal 32.

## Exacte controles

- Capture-routes en routergewichten: `True`.
- Rank/count- en expert-count-cancellationcontroles: `True`.
- Full-rank operatorimages (`x/g/u/z/y`): `True`.
- Opgeslagen BF16-`z`, opnieuw gegroepeerd met andere GEMM-batchvorm, bit-exact (diagnostisch, geen gate): `False`.

## Claimgrens

Dit is een rank- en page-faultscreen. De koude bytes zijn analytische packed-int4-boekhouding; atlasreads, projectiecompute, kwaliteit, latency en SSD-stalls zijn nog niet gemeten. Een positief P1-resultaat is daarom geen Eureka en geen runtimeclaim.

## Volgende actie

Sluit P1 op V2. Een hogere-E-proef vereist een afzonderlijke preregistratie; dit resultaat mag niet post-hoc met een andere threshold worden gered.
