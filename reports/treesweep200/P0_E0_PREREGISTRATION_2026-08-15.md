# P0 + E0 — identiteit, baseline en N1–N5-reproductie: preregistratie

Datum: 2026-08-15
Registry: `reports/treesweep200/EXPERIMENT_REGISTRY.yaml` (NEMOTRON_TREESWEEP_200_ROOFLINE_V2)
Status: **bevroren vóór uitvoering.**
Agent-specs: `agents/01_IDENTITY_BASELINE.md`, `agents/18_ROOFLINE_REPRODUCTION.md`
uit `info/NEMOTRON_TREESWEEP_200_ROOFLINE_V2_AGENT_PACK_2026-08-15`.

## 1. Wat deze fase is

P0 legt de identiteit van het target vast en reproduceert de bevroren baseline.
E0 importeert de vijf N1–N5-metingen als `USER_MEASURED_UNVERIFIED`, hasht hun
ruwe bronnen, en classificeert elk als `reproduced`, `shifted`, `invalid` of
`inconclusive` — het verdict-woordenschat van het contract.

Deze fase bouwt niets en meet niets nieuws op de GPU behalve één goedkope
reproductie: de streaming-roofline (N5). Alle overige grootheden bestaan al als
onafhankelijk geverifieerde artefacten in de LIGHTNINGSTREAM_NEMOTRON-registry;
E0 hasht die bronnen en herberekent de afgeleide getallen (bytevloeren,
percentages) onafhankelijk, uit de ruwe data en uit de safetensors-headers van
het checkpoint — niet uit de conclusies van de eerdere rapporten.

## 2. Identiteitsbronnen (alle gelezen, gehasht, niet gewijzigd)

- modelmap `models/nemotron_3_5_lightning_v35` (config, tokenizer, index);
- `reports/lightningstream_nemotron/n2r_v35_layout.json` (layout-lock van het
  v35-checkpoint);
- `reports/lightningstream_nemotron/s10a_wiring_resolution.json` (MTP-wiring,
  empirisch vastgesteld uit vier kandidaten met NLL-beslissing);
- runtime: `src/moe_lab/lightningstream_nemotron/runtime.py` en de modules die
  hij importeert.

## 3. Evidence-import (bronnen van de vijf claims)

| claim | bronartefact |
|---|---|
| N5 roofline 338,4 GB/s + bytevloeren | `n1n2n4n5_ceilings.json` |
| N1 graph-winst 23,7% | idem |
| N2 gather 8,192 ms / 4,3 GB/s | idem |
| N4 attention 47,2 GB/s, fit R²=0,9964 | idem |
| N5 GEMV 81,4 GB/s | `y2r1_bytes_vs_time.json` |
| N3 ReLU²-prefilter gesloten | `n3_relu2_prefilter_oracle.json` |
| bestaande verificatie | `n1_n5_independent_verification.json` (36/36) |

## 4. Wat E0 zelf opnieuw doet

1. **Bytevloeren**: compulsory bytes per token bij ctx 0 / 4K / 32K / 128K /
   262.100 herberekend uit de safetensors-headers van het checkpoint (welke
   tensors per token gelezen moeten worden: gepinde expert-records via de
   gemeten hitrate, KV-cache-inhoud, residente shell die per token gelezen
   wordt) — een eigen telling, geen overname van N5's tabel.
2. **Roofline**: de streaming-leesbandbreedte opnieuw gemeten met een eigen
   kernel (256 MiB, `float4`-loads), warme klokken, 10 herhalingen.
3. **Classificatie** per geïmporteerde claim: `reproduced` als E0's eigen
   waarde binnen 10% van de geïmporteerde ligt of de claim uit een reeds
   geverifieerd in-repo artefact volgt; anders `shifted`/`invalid`/
   `inconclusive` met onderbouwing.

## 5. Poorten

- **G-P0-I1 — identiteit**: model-ID, revisie, tokenizer, laagpatroon,
  expert-aantallen, top-k, hidden, Mamba/GQA-configuratie, kwantisatie en
  MTP-structuur vastgelegd in `P0_IDENTITY_MANIFEST.json`, alle bronnen
  gehasht. Zonder volledige manifest opent geen latere tak.
- **G-P0-B1 — baseline**: de bevroren baseline ligt in
  `n7b_cached_decode.json` (27,743/26,200/21,699/18,424 tok/s); E0 hoeft hem
  niet opnieuw te draaien maar documenteert hash en herkomst. (Deze baseline is
  in S10A al gereproduceerd binnen 2,6%.)
- **G-E0-R1 — roofline**: eigen streaming-meting binnen 10% van 338,4 GB/s,
  anders classificatie `shifted` met de eigen waarde als nieuwe referentie.
- **G-E0-F1 — vloeren**: E0's eigen bytevloeren binnen 10% van 6,05 ms (ctx 0)
  en 8,43 ms (262K) bij de gemeten roofline, of `shifted` met eigen getallen.

## 6. Wat deze fase niet doet

Geen optimalisatie, geen boomverifier, geen drafter, geen tok/s-claim. De
TreeSweep-as is al gesloten door Z1 (plafond 45–61 tok/s tegen poort 250) en
wordt hier niet heropend; deze fase legt alleen het fundament voor de
exact-efficiency track E1–E6.

## 7. Verificatie

Onafhankelijke verifier (`e0_independent_verify.py`) importeert de runner niet,
herberekent de bytevloeren uit de checkpoint-headers met een eigen telling,
controleert alle hashes tegen deze preregistratie en evalueert de poorten
opnieuw.

## 8. Artefacten (te produceren)

`scripts/treesweep200/p0e0_identity_roofline.py` ·
`reports/treesweep200/P0_IDENTITY_MANIFEST.json` ·
`reports/treesweep200/E0_N1_N5_EVIDENCE_MANIFEST.json` ·
`reports/treesweep200/E0_ROOFLINE_REPRODUCTION.json` ·
`scripts/treesweep200/e0_independent_verify.py` ·
`reports/treesweep200/e0_independent_verification.json` ·
`reports/treesweep200/P0_E0_REPORT_2026-08-15.md`
