# S100 Phase 16R — repaired localization, horizon and Mamba scan

Datum: 2026-08-19  
Pack SHA256: `c08da896eddd2cbd486c5099060af8a05fa9e09aacd2cf21ff5f3f44ef6403a0`  
Branch: `agent/s100-phase16R-repair`

## Preflight

De oorspronkelijke 16R-pack faalde nog op importvolgorde: CuPy werd vóór Torch
geladen. De bewezen Phase-14 D2-test laadt Torch eerst. Met deze minimale
harness-repair werd de preflight groen:

- q-projection DLPack → BF16 → clone → transpose → contiguous → MM: finite;
- q-projection-vorm: `2688 × 4096`;
- SSM-state: `524288 = 64 × 64 × 128`;
- output `y`: `4096 = 64 × 64`;
- `PREFLIGHT_GREEN: true`.

De inhoudelijke run is daarna volledig uitgevoerd.

## 16A — local sensitivity

Alle 37 BF16-matrices en de families zijn vanuit exacte state getest. Er was
geen strict-green individuele matrix, dus `safe_matrix_count = 0` en 16B had
geen subset om te valideren.

Sterkste lokale voorbeelden waren attention-K/V-matrices, maar ook die waren
niet strict groen. De beste losse matrix was ongeveer `attention_26_v` met
98,75% top-1 agreement; meerdere K-matrices lagen rond 97,5%. De belangrijkste
families:

- attention-K: 92,5% top-1, K16 100%, mean KL 0,0189;
- attention-V: 90,0% top-1, K16 100%, mean KL 0,0610;
- Mamba-out: 87,5% top-1, K16 100%, mean KL 0,1935;
- Mamba-in: 10,0% top-1, K16 30%, mean KL 6,4966.

De drift lokaliseert daarmee opnieuw vooral in Mamba-in en in de collectieve
state-interactie; losse attention-K/V-matrices zijn relatief mild maar halen
de vooraf ingestelde strict gate niet.

## 16C — exact-state horizon

De geldige horizonmeting op de beschikbare `attention_all`-arm gaf 40 blocks
per horizon:

| Horizon | Eerste voorspelling gelijk | Gemiddelde exacte prefix | Volledig gelijk block |
|---:|---:|---:|---:|
| H=1 | 77,5% | 0,775 | 77,5% |
| H=2 | 77,5% | 1,325 | 55,0% |
| H=4 | 70,0% | 1,775 | 20,0% |
| H=8 | 77,5% | 2,200 | 5,0% |

`ANY_H4_BLOCK_RESEARCH_GO` bleef false. Dit is nu een echte exact-state
meting, geen technische failure, maar de geselecteerde attention-all route
haalt de H4-gate niet.

## 16E — affine Mamba prefix scan

Dit onderdeel is inhoudelijk groen. Op vroege, middelste en late echte Mamba-
lagen (`0`, `25`, `50`) werden de productie-SSM-update en een H=8
Hillis-Steele-prefix-scan vergeleken.

- state-elementen: `524288`, exact gelijk aan `64 × 64 × 128`;
- maximale formule-B NRMSE: `8,33e-7`;
- scan versus GPU-state NRMSE: maximaal `1,09e-7`;
- scan versus GPU-output `y`: maximaal `2,99e-7`;
- alle waarden liggen ruim onder de scan-tolerantie `5e-5`;
- alle drie lagen: `pass: true`.

Daarmee is voor deze echte SSM-data aangetoond dat de state-recurrence
algebraïsch naar een parallel affine prefix-scan kan worden omgeschreven. Dit
is een structureel resultaat, maar nog geen gemeten CUDA-microkernel of
throughputclaim.

## Eindadjudicatie

```text
Instrumentation complete: True
Safe matrix count: 0
Selected strict subset: None
ANY_H4_BLOCK_RESEARCH_GO: False
MAMBA_AFFINE_SCAN_BUILD_OPEN: True
SAFE_BF16_SUBSET_RUNTIME_BUILD_OPEN: False
PARALLEL_BLOCK_VERIFIER_RESEARCH_OPEN: True
NEXT_ROUTE: BUILD_MAMBA_SCAN_MICROKERNEL_THEN_RETEST_BLOCK
S100 SINGLE ACHIEVED: False
```

De juiste vervolgstap is dus Phase 17: een echte parallelle Mamba-scan-
microkernel bouwen en daarna de B=4 block-verifier opnieuw meten. Native BF16
als algemene drop-in blijft gesloten; de affine Mamba-scanroute is wél geopend
voor verder onderzoek.
