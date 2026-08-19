# S100 Phase 15 — Native BF16 Fidelity Repair run report

Datum: 2026-08-19  
Pack SHA256: `10c227637bb54b169bc2ab24fee6326504d1c0f639d290cc749b6999267837de`  
Branch: `agent/s100-phase15-native-bf16-fidelity`

## Integriteit

De ZIP-hash en het meegeleverde manifest zijn gecontroleerd. De runner is
uitgevoerd zonder `-SkipMatrixSensitivity`, met dezelfde vijf-shard
Nemotron-checkpoint als Phase 14.

## Gecorrigeerde teacher-forced fidelity

Phase 15B voerde na ieder meetpunt de exacte parent-token aan de candidate
door. De candidate liep dus niet meer op zijn eigen greedy trajectory.

| Arm | Validation top-1 | K=16 inclusion | Strict pass |
|---|---:|---:|---|
| `mm_fp32out/all` | 1.67% | 10.42% | nee |
| `mm_fp32out/attention` | 56.67% | 85.42% | nee |
| `mm_fp32out/mamba` | 24.58% | 55.42% | nee |
| `mm_fp32out_comp2/all` | 1.25% | 10.42% | nee |

De nieuwe K2-protocolmeting is daarmee geldig. Zij corrigeert de oude
0,625%-meting als bewijsstuk, maar redt de volledige native-BF16-substitutie
niet: de `all`-arm blijft zeer ver van exacte fidelity. Attention-only is een
duidelijk betere lokale kandidaat, maar voldoet nog niet aan de strikte gate.

## Technische onderdelen

- 15A component-contracttest: technisch afgebroken op
  `CUBLAS_STATUS_INVALID_VALUE` bij `torch.mm(x_bf16, W^T,
  out_dtype=torch.float32)`. Daardoor zijn B=1/B=4-speedups en comp2-speedup
  niet beschikbaar uit deze run.
- 15C matrix sensitivity: technisch afgebroken door
  `cudaErrorMemoryAllocation` bij de pinned-memory allocatie van de routed
  bank. `matrix_safe_count` is dus onbekend, niet nul.
- 15D exact-state horizons H=1/2/4/8: beide varianten liepen technisch vast
  op dezelfde CUBLAS-layoutfout tijdens de native Mamba-step. Er is daarom
  geen geldig H4-resultaat.
- Heldout is niet geopend, omdat geen validation-arm strict groen was.

## Adjudicatie

```text
DIRECT_NATIVE_BF16_RUNTIME_BUILD_OPEN: False
EXACT_STATE_BLOCK_DRAFT_BUILD_OPEN: False
SELECTIVE_MATRIX_RUNTIME_RESEARCH_OPEN: False
NEXT_ROUTE: LOCALIZE_OR_CLOSE_NATIVE_BF16_DIRECT_SUBSTITUTION
S100 SINGLE ACHIEVED: False
```

De juiste tussenconclusie is dus tweedelig: de Phase-14 K2-uitkomst mag niet
als definitief protocolbewijs worden gebruikt, maar de gerepareerde Phase-15B
teacher-forced test geeft nog steeds geen groen bewijs voor volledige native
BF16. Attention-only verdient eventueel gerichte vervolgmeting; de matrix- en
horizonresultaten moeten eerst technisch reproduceerbaar worden gemaakt voordat
die routes definitief kunnen worden gesloten.
