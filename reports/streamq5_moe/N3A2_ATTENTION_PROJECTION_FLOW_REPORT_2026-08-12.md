# N3A2 — exacte attention projection-flow-fusie

Datum: 2026-08-12. Status: **fysieke componentpass**.

## Uitkomst

Eén aaneengesloten ERVF-16-grid voor Q, K en V is sneller dan drie afzonderlijke
projectielaunches, met behoud van alle bestaande BF16-grenzen. De geselecteerde
`concat_qkv`-arm laat de gedeelde input-RMSNorm gematerialiseerd en gebruikt
daarna de ongewijzigde Q/K-norm+RoPE- en V-KV-writekernels.

| Fase | baseline | `concat_qkv` | ratio |
|---|---:|---:|---:|
| validation p50 | 4,3864 ms | 3,8842 ms | 0,8855 |
| test p50 | 4,7466 ms | 4,1852 ms | 0,8817 |
| test p95 | 5,6916 ms | 5,0630 ms | 0,8896 |

De test-p50-speedup is **1,1342×**. De vooraf vastgelegde grenzen van 0,97 p50
en 1,00 p95 slagen beide.

## Exactheid

Beide kandidaten waren bitexact op validation:

- 245.760 FP32 Q/K/V-uitgangen per arm, nul afwijkingen;
- 49.152 geschreven BF16 K/V-cache-elementen, nul afwijkingen;
- alle waarden eindig.

De geselecteerde arm was opnieuw volledig bitexact op de afgesloten
testpositie 3079.

De preregistratietekst noemt abusievelijk 294.912 FP32-uitgangen. De expliciete
scope was alle Q/K/V-tensoren; hun werkelijke omvang is
`48 × (4096 + 512 + 512) = 245.760`. Er is geen tensor of laag overgeslagen.
Dit is een rekenfout in de beschrijving, niet een wijziging van kandidaat,
partitie, selectie of meetpoort; het preregistratiebestand is na de run niet
aangepast en de oorspronkelijke hash blijft controleerbaar.

## Negatieve controle: `head_flow`

De agressievere projectie→Q/K-norm→RoPE→KV-writefusie was eveneens bitexact,
maar validation-p50 bedroeg 9,0867 ms tegenover 4,3864 ms voor de baseline.
Eén block per head moet acht projectiegolven serieel uitvoeren; de sterk lagere
inter-head/projectierij-paralleliteit kost meer dan vier uitgespaarde launches.
Deze head-blockgeometrie is daarom gesloten.

## Wat bewezen is

Voor deze fysieke 48-laagse Q8-bank is Q/K/V **launch-concatenatie** een nuttige
exacte optimalisatie. Het resultaat bewijst niet dat de Q8-gewichten of
activaties onderling gedeeld zijn: alleen de bestaande gedeelde RMSNorm-input
en de drie logische rijruimtes zijn in één grid samengebracht.

Attention scores/values, expertpad, H2D, volledige decodertijd, kwaliteit,
andere GPU's en externe SOTA vallen buiten de claim.

Auditspoor:

- `N3A2_ATTENTION_PROJECTION_FLOW_PREREGISTRATION.md`;
- `scripts/streamq5_moe/run_n3a2_attention_projection_flow.py`;
- `n3a2_attention_projection_flow.json`;
- `scripts/streamq5_moe/verify_n3a2_attention_projection_flow.py`;
- `n3a2_attention_projection_flow_verification.json`.
