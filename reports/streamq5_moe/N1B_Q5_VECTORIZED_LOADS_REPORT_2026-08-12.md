# N1B — Q5 aligned-word loads

Datum: 2026-08-12. Status: **geïsoleerde componentpass**.

## Resultaat

De geselecteerde `aligned32x2`-variant is bitexact over alle 1.376.256
gate/up/down-uitvoerelementen van de 48-laagse Q5-plane en versnelt die plane
op de aanwezige RTX PRO 2000 Blackwell Laptop GPU.

| Metriek | bestaande ERVF-16 | `aligned32x2` | ratio |
|---|---:|---:|---:|
| test p50 | 7,8479 ms | 6,9673 ms | 0,8878 |
| test p95 | 10,7937 ms | 9,3385 ms | 0,8652 |

De p50-speedup is **1,1264×**. De vooraf vastgelegde grenzen waren p50-ratio
`<=0,97` en p95-ratio `<=1,00`; beide slagen. De 120 testparen werden in
afwisselende AB/BA-volgorde gemeten.

Beide kandidaten waren bitexact. Op validation waren de p50-tijden:

- baseline: 7,4667 ms;
- `aligned64x2`: 6,6371 ms;
- `aligned32x2`: 6,4841 ms.

## Wat dit technisch zegt

De vijf bytes per 40-bit Q5-pack afzonderlijk samenstellen is op deze kernel
niet optimaal. Twee natuurlijk uitgelijnde 32-bit vensters lezen, verschuiven
en maskeren verlaagt de volledige fysieke projectieplanetijd zonder de
Q5-codes, schalen, BF16-afronding of reductievolgorde te veranderen.

De bronvorm gebruikt 32-bit loads. Zonder een lokale CUDA-toolkit/disassembler
is niet afzonderlijk vastgelegd welke exacte SASS-loadinstructies NVRTC heeft
uitgegeven; de reproduceerbare fysieke timing en bitexactheid zijn wel direct
gemeten.

## Auditspoor en grens

- preregistratie: `N1B_Q5_VECTORIZED_LOADS_PREREGISTRATION.md`;
- evaluator: `scripts/streamq5_moe/run_n1b_q5_vectorized_loads.py`;
- ruwe uitvoer: `n1b_q5_vectorized_loads.json`;
- preregistratiehash in uitvoer komt exact overeen met het huidige bestand;
- twee compile-only pogingen met `memcpy` stopten vóór correctness of timing
  doordat de lokale NVRTC geen C-headers/builtin bood; de vervanging is vóór
  het openen van experimentele resultaten in de preregistratie genoteerd.

Dit is geen end-to-end-pass. Integratie in de decoder, een omgekeerde
replicatie buiten de gepaarde volgorde en een eventuele SASS-audit blijven
aparte vervolgstappen. Het resultaat bewijst evenmin generalisatie naar andere
GPU's of SOTA ten opzichte van externe runtimes.
