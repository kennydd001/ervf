# D1 — deterministische accumulatievolgorde: opgelost, en NERVF-5 slaagt alsnog

Datum: 2026-08-15 · Namespace `NERVF_NEMOTRON`
Verdict: **Met accumulatie in routevolgorde is de runtime run-to-run deterministisch: `base_b` is nu identiek aan `base_a` over 3 x 512 tokens. Daarmee slaagt NERVF-5 op alle drie zijn poorten, inclusief de exactheidspoort die eerder niet evalueerbaar was.**
Terminal state: `d1_route_order_accumulation_restores_determinism`

## Wat er veranderd is

NERVF-5 legde bloot dat `_moe_cached` de zes routed experts in
**hit-dan-miss-volgorde** accumuleert. Welke expert een hit is hangt van de
LRU-staat af, dus twee runs met andere cachegeschiedenis telden in andere
volgorde op — en FP-optelling is niet associatief.

De ingreep scheidt **rekenvolgorde** van **optelvolgorde**:

- rekenen blijft in hit-dan-miss-volgorde, dus de latencywinst (hits rekenen
  terwijl misses binnenkomen) blijft volledig behouden;
- elke expertbijdrage gaat naar zijn eigen slotbuffer;
- na de lus worden de zes bijdragen in **routevolgorde** (`s = 0..5`) opgeteld.

Kosten: `top_k x hidden` floats per runtime (64 KB), en nul extra kernels — het
aantal `accumulate_into`-aanroepen blijft zes. Opt-in via
`rt.deterministic_accum`, default **uit**, zodat elke eerdere meting het pad
blijft beschrijven dat zij gemeten heeft.

## Uitkomst, 3 domeinen x 512 causale tokens

| poort | zonder D1 | met D1 |
|---|:--|:--|
| `base_b` identiek aan `base_a` | ❌ | **✅** |
| **G-NERVF-5C** exactheid | ❌ niet evalueerbaar | **✅** |
| **G-NERVF-5P** latency | ❌ | **✅** |
| **G-NERVF-5M** VRAM | ✅ | ✅ |

| domein | basis p50 | ERVF p50 | winst | drift | conclusief |
|---|---:|---:|---:|---:|:--:|
| expository | 42,507 | 39,736 | **+2,771** | 0,361 | ✅ |
| narrative | 43,659 | 38,650 | **+5,008** | 2,013 | ✅ |
| code | 42,878 | 37,483 | **+5,395** | 1,121 | ✅ |

**ERVF is nu bewezen exact over 512-token rollouts**, niet alleen over 2 x 64,
en de winst is in elk domein conclusief.

## Waarom dit breder telt

Elke exactheidspoort in deze lijn en in Kimi's E-lijn vergeleek twee armen van
een runtime waarvan de uitvoer van de cachegeschiedenis afhing. Die poorten zijn
waar voor de runs waarin ze gemeten zijn, maar ze konden de eigenschap niet
gàranderen. Met D1 aan kan dat wel, en dat is een voorwaarde voor elke latere
integratie (E6) waar meerdere wijzigingen tegelijk gegate worden.

Aanbeveling: `deterministic_accum` default aanzetten zodra een fase het als
basislijn gebruikt. Ik heb de default niet omgezet — dat is een
productiebeslissing.

## Claim boundary

512-token causale rollouts op deze GPU bij capacity 72, drie domeinen, drie armen
base/ervf/base. Determinisme is vastgesteld door **twee armen met identieke
configuratie** te vergelijken, dus onafhankelijk van ERVF. Latency is end-to-end
wandtijd per token inclusief synchronisatie; de context groeit tijdens de
rollout, dus deze getallen zijn niet vergelijkbaar met n7b's vaste-diepte-cijfers.

## Artefacten

`src/moe_lab/lightningstream_nemotron/runtime.py` (`deterministic_accum`,
default uit) · `scripts/nervf_nemotron/nervf5_full_model.py` (D1-vlag) ·
`d1_determinism.json` · `nervf5_full_model.json` (de eerdere, gefaalde run)
