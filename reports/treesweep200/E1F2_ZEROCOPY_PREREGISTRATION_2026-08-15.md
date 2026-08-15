# E1 fase 2 — graph-resident token — preregistratie microbench (2026-08-15)

Bevroren vóór elke meting van deze fase. Poorten worden na het zien van
resultaten niet verruimd. Voortbouwend op `E1_GRAPH_ORACLE_REPORT_2026-08-15.md`
(fase 1: budget 8,925 ms/token bij ctx 64, mét ERVF) en de V1/E2-weerlegging
(strided host-reads: 6,7 GB/s vs 85,9 GB/s device).

## Hypothese (het nieuwe ontwerp)

De enige reden dat de tokenlus naar de host synchroniseert is dat de **host**
beslist welke experts gemist worden en de DMA-kopie uitgeeft. Maar de pinned
bank is al mapped (productiecode leest down_proj-kolommen er rechtstreeks uit
via `down_base_ptr`). Als een kernel een miss-expert **zelf bulk-leest uit de
pinned bank** (contiguous rijen, niet strided kolommen — dat laatste is wat V1
sloot), dan kan routes-selectie, cache-raadpleging én miss-afhandeling
device-side, en wordt de hele token graph-captureerbaar.

Dat staat of valt met een ongemeten grootheid: **bulk zero-copy
leesbandbreedte vanuit een kernel over PCIe**. V1 mat 6,7 GB/s *strided*; de
DMA-engine haalt 26,03 GB/s. Niemand heeft op deze machine gemeten wat een
kernel haalt bij het lezen van een hele expert-record (2,6 MiB, rij-majeur).

## Accounting (bevroren, bepaalt de kill-gate)

Per token (huidige up_only-cache, 65% hitrate): 23 MoE-lagen × 6 experts × 35%
miss ≈ 48 missen × 2,6 MiB ≈ **127 MB miss-bytes**. Via DMA: 127/26,03 =
4,9 ms. Graph-residentie bespaart (fase 1 + S14/Y1, gemeten): uitgifte
8,9 ms + route-readback/host-gap ≥ 4,7 ms ≈ **13,6 ms**. Breakeven:
127 MB / B = 4,9 + 13,6 → B ≈ 6,9 GB/s.

## Metingen (runner `scripts/treesweep200/e1f2_zerocopy_microbench.py`)

- **M0** correctheid: kernel-checksum over een pinned host-buffer met bekende
  inhoud (UVA-mapping werkt en leest correct).
- **M1** streaming bulk-read: pinned pool 256 MB, grid-stride uint4-checksum,
  GB/s. Zelfde kernel op een device-kopie als referentie.
- **M2** beslissend: de echte productie-ERVF `gemv_into` (rows=1856,
  cols=2688, apply_relu2) op een synthetische NVFP4-expert-record, gewichten
  eenmalig op device vs gemapped op host, cyclend over 24 distincte records
  (62 MB pool — voorkomt L2-artefacten, NERVF-1-les). 50 reps, mediaan.
  Effectieve GB/s = (UP_CODE + UP_SCALE) / t.
- GPU-vrijheidscontrole vooraf (`nvidia-smi --query-compute-apps`), exit 4 bij
  bezetting.

## Poorten (bevroren)

- **G-E1F2-M0**: host-read checksum exact gelijk aan de host-berekende waarde.
- **G-E1F2-M2X**: M2 host-arm en device-arm produceren **bitidentieke** output
  op alle 24 records (zelfde bytes, zelfde kernel — enige variabele: waar het
  gewicht fysiek staat).
- **G-E1F2-K1** (kill/proceed op M2 host-arm effectieve bandbreedte):
  - ≥ 12 GB/s → **bouwen** (missen ≤ 2,2× DMA-tijd, ruim binnen breakeven);
  - 6,9–12 GB/s → **marginaal**: alleen bouwen als de whole-token-accounting in
    het rapport netto positief blijft;
  - < 6,9 GB/s → **stop**: netto negatief onder de bevroren accounting; E1
    fase 2 via dit pad weerlegd, documenteren en door.

## Beslisregels

- M2X faalt → geen enkele verdere conclusie; eerst de mapping/reproductie
  repareren. Geen tolerantie verruimen.
- K1 < 6,9 → E1-fase-2 via in-kernel zero-copy **weerlegd**; dat sluit
  graph-residentie niet (alleen dit transport), maar het alternatief moet dan
  een nieuw idee zijn.
- Alle getallen zijn componentmetingen; geen tok/s-claims. De accounting is
  een model met expliciet gemeten inputs, geen meting van een runtime.
