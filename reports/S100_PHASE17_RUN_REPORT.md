# S100 Phase 17 — Real Mamba block scan run report

Datum: 2026-08-19  
Pack SHA256: `4f72771abc607a463d6fe5ce2f53d523db88d45e2dd83515a41ec78867a13368`  
Branch: `agent/s100-phase17-mamba-block-scan`

## Preflight

De ZIP-hash en het manifest zijn gecontroleerd. De CUDA preflight was groen op
de NVIDIA RTX PRO 2000 Blackwell Laptop GPU:

- prefix synthetic NRMSE: `5,74e-8`;
- serial synthetic NRMSE: `5,57e-8`;
- `PREFLIGHT_GREEN: true`.

## H=4 op echte Mamba-lagen

Alle metingen bleven binnen de strikte state/output-NRMSE-gates. De serial
variant was voor H=4 op alle drie lagen de snelste SSM-keuze.

| Laag | SSM speedup | Core speedup | Volledige laag | Full-layer NRMSE |
|---:|---:|---:|---:|---:|
| 0 | 1,671× | 2,176× | 1,096× | 8,16e-8 |
| 25 | 2,053× | 3,429× | 1,102× | 2,18e-7 |
| 50 | 1,771× | 2,394× | 1,092× | 9,62e-8 |

De SSM- en core-gates zijn daarmee overtuigend groen. De volledige laag zit
echter rond de 1,10×-grens: laag 25 haalt hem nipt, lagen 0 en 50 blijven er
net onder. De bestaande exacte in/out-projecties zijn dus de resterende
bottleneck.

Bij H=8 blijft het patroon zichtbaar: de core blijft sterk, maar de volledige
laag varieert ongeveer van 0,98× tot 1,18× afhankelijk van de laag. Dit is nog
geen stabiele B=4-verifier-economie.

## Adjudicatie

```text
Instrumentation complete: True
SSM_SCAN_MICROKERNEL_OPEN: True
MAMBA_CORE_BLOCK_OPEN: True
MAMBA_LAYER_B4_CEILING_OPEN: False
PHASE18_FULL_BLOCK_VERIFIER_OPEN: False
NEXT_ROUTE: OPTIMIZE_PROJECTION_BLOCKING_THEN_RETEST_FULL_LAYER
S100 SINGLE ACHIEVED: False
```

Dit is een echte structurele winst op recurrence- en core-niveau, maar nog geen
volledige block-verifier. De juiste vervolgstap is projection blocking/
format-preserving projectie-optimalisatie, daarna dezelfde volledige-laagtest
opnieuw. Er is nog geen end-to-end tok/s- of S100-claim.
