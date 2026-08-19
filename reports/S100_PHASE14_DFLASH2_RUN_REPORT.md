# S100 Phase 14 DFlash2 — run report

Datum: 2026-08-19  
Pack SHA256: `aa212902bba8668476cb97b135e6f073dcfc430be3fe5d017c736f1e434020c4`  
Branch: `agent/s100-phase14-dflash2-hardware`

## Integriteit en uitvoering

De ZIP-hash en het meegeleverde manifest zijn gecontroleerd. De volledige
runner is uitgevoerd met de cached vijf-shard Nemotron-checkpoint. De runner
heeft fail-soft gepubliceerd; drie oudere Phase-14-onderdelen waren technisch
onvolledig, maar 14F0 en 14F1 zijn volledig gemeten.

## 14F0 — verifier-economie en resident geheugen

De bestaande bit-exacte verifier blijft de harde bottleneck:

| Block | Gemeten verifiercyclus | Perfect-draft plafond |
|---:|---:|---:|
| B=2 | 35,56 ms | 56,25 tok/s |
| B=4 | 70,99 ms | 56,34 tok/s |
| B=8 | 143,90 ms | 55,59 tok/s |

Zelfs een perfecte drafter kan deze verifier dus niet naar 100 tok/s brengen.
De B=4 Phase-12C-waarde van 60,94 ms is alleen een projectie en geeft ongeveer
65,64 tok/s, geen gemeten geïntegreerde runtime.

De resident-memory gate is eveneens gesloten. Na de quality-parent bleef op de
RTX PRO 2000 Blackwell Laptop GPU geen vrije VRAM over. Zelfs de kleinste
gescreende kandidaat — twee NVFP4-draftlagen, MLP-ratio 2,0 — vraagt voor 4K
context ongeveer 664,7 MiB extra met BF16 draft-KV en past niet in de gemeten
resident envelope.

## 14F1 — suffix-decay correction

De echte Nemotron calibration/validation proxy gebruikte rank 64, group size
128 en top-K 16. Op validation veranderde de correction de hidden-state error
nauwelijks en zelfs licht negatief:

- mean NRMSE: `0,65694` → `0,65782`;
- laatste drie suffixposities: `0,67498` → `0,67735`;
- gemiddelde top-1: `22,05%` → `22,10%`;
- top-K recall: `34,51%` → `34,55%`;
- onafhankelijke acceptance inclusief anchor: `2,4625` → `2,4656` tokens.

Dit is geen voldoende suffix-decay-signaal. De correction wordt daarom niet als
trainingsroute geopend.

## 14F1 — parallel path selection

De candidate lattice had een oracle-bovengrens van gemiddeld `2,75` tokens
inclusief anchor. De frozen selector-proxy haalde `2,46875` tokens tegenover
`2,465625` zonder selector: winst `0,003125` token, slechts ongeveer 1,1% van
de beschikbare oracle-headroom. Dit is geen overtuigend transition-scorer-
signaal en rechtvaardigt geen DFlash2-training.

## Overige Phase-14-resultaten

14D mat 31 BF16-matrices, maar de componentgate bleef false bij B=2, B=4 en
B=8. De afzonderlijke matrix-sommen waren respectievelijk ongeveer 0,12×,
0,15× en 0,44× native versus de huidige ERVF-baseline; dit is geen end-to-end
throughputclaim. Officiële quality validation ontbrak opnieuw wegens de
ontbrekende Phase-3/4-traces.

14B2 faalde technisch omdat de huidige runtime vanuit `_moe` `None` teruggaf;
14E2 kon daardoor niet starten omdat de 14B2-capture ontbrak. Deze twee zijn
technische gaten, geen inhoudelijke falsificatie van DFlash2.

## Eindadjudicatie

```text
DFLASH2_CURRENT_VERIFIER_S100_OPEN: false
DFLASH2_RESIDENT_MEMORY_OPEN_4K: false
DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN: false
DFLASH2_TRAINING_BUILD_OPEN: false
S100 SINGLE ACHIEVED: false
```

Besluit: niet trainen vanuit deze goedkope proxy-evidence. De twee ideeën zijn
op echte Nemotron-states getest; de suffix-correction levert geen bruikbaar
signaal en de selector benut vrijwel geen lattice-headroom. Een volgende
DFlash2-iteratie heeft eerst een echte multi-row verifier onder ongeveer 10×B
ms en een resident-memory-oplossing nodig.
