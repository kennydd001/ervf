# S100 Phase 16 — Localize, Horizon and Mamba Scan run report

Datum: 2026-08-19  
Pack SHA256: `3eea9143b15b9bac376b4ff888f43e064c060e42b011274cb9a049471c4a28b5`  
Branch: `agent/s100-phase16-localize-horizon-scan`

## Integriteit en uitvoering

De ZIP-hash en het meegeleverde manifest zijn gecontroleerd. De volledige
runner is uitgevoerd met dezelfde cached vijf-shard Nemotron-checkpoint.

## Resultaten

- 16A local sensitivity: technisch afgebroken vóór een matrixresultaat. De
  eerste native transition faalde in `attention q_proj` op
  `CUBLAS_STATUS_INVALID_VALUE`. `safe_matrix_count` is daarom onbekend, niet
  nul.
- 16B subset validation: er werden geen subsets getest omdat 16A geen veilige
  matrixrangschikking opleverde. Het opgeslagen `safe_matrix_count: 0` is een
  dependency-uitkomst, geen wetenschappelijke falsificatie.
- 16C exact-state horizon: technisch afgebroken op dezelfde CUBLAS-fout in
  `q_proj`, vóór geldige H=1/2/4/8-records. `ANY_H4_BLOCK_RESEARCH_GO` is dus
  onbekend; de pack-summary zet hem fail-soft op false.
- 16D savings: geen geselecteerde matrices, dus geen geldige besparing.
- 16E affine Mamba scan: technisch afgebroken vóór de scanvergelijking. De
  CPU-reconstructie probeert een state-array van 4096 elementen te reshapen
  naar `(64, 64, 128)`. `MAMBA_AFFINE_SCAN_BUILD_OPEN` is daarom onbekend; de
  algebraïsche hypothese is niet getest.

## Adjudicatie

De runner publiceerde:

```text
Instrumentation complete: False
Safe matrix count: None
Selected strict subset: None
ANY_H4_BLOCK_RESEARCH_GO: False
MAMBA_AFFINE_SCAN_BUILD_OPEN: False
SAFE_BF16_SUBSET_RUNTIME_BUILD_OPEN: False
PARALLEL_BLOCK_VERIFIER_RESEARCH_OPEN: False
NEXT_ROUTE: CLOSE_NATIVE_BF16_SUBSTITUTION_KEEP_SCAN_RESEARCH
S100 SINGLE ACHIEVED: False
```

Omdat `instrumentation_complete` false is, mogen de false-flags voor 16A,
16C en 16E niet als inhoudelijke sluiting worden gelezen. Phase 16 heeft de
matrixlokalisatie, horizonmeting en affine-scan nog niet daadwerkelijk kunnen
beoordelen. De volgende reparatie moet eerst de native `q_proj`-dispatch
reproduceerbaar maken en de Mamba-statevorm uit de echte records afleiden,
waarna 16A/16C/16E opnieuw kunnen worden uitgevoerd.
