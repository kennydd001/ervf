# NERVF-3 — ERVF in de echte runtime: exact, en −3,7 tot −4,5 ms per token

Datum: 2026-08-15
Namespace: `NERVF_NEMOTRON`
Verdict: **Exactheid volledig: alle drie de armen produceren identieke tokens, óók tegen het bevroren V35-anker. De tokenwinst is op alle drie de contextdiepten conclusief: −3,701 ms bij ctx 0, −3,102 bij 131K, −4,505 bij 262100. De componentpoort G-NERVF-3P faalt (1,144× tegen 1,35×), maar meet een venster dat veel meer bevat dan wat ERVF vervangt.**
Terminal state: `nervf3_integrated_exact_token_gain_conclusive_component_gate_diluted`
Vorige fase: `NERVF_1_2_REPORT_2026-08-15.md` (1,936× bitexact op het projectievlak)

## 1. Wat er geïntegreerd is

De ERVF-kernel (breedte 16) staat nu **additief** in
`src/moe_lab/lightningstream_nemotron/fused_nvfp4.py`, achter `use_ervf` met
default **False**, zodat elke eerdere meting het pad blijft beschrijven dat zij
gemeten heeft. Aangezet vervangt hij `gemv_nvfp4_rows` overal waar die wordt
aangeroepen: de routed-expert up-projectie, beide shared-expert-projecties, de
NVFP4-Mamba-projecties en de LM-kop.

Drie armen, `base / ervf / base`, één modelload, twee rondes van 16 gemeten
stappen per context. Enige variabele: `use_ervf`.

## 2. Exactheid — de harde poort

| arm | pariteit tussen armen | pariteit tegen het V35-anker |
|---|:--:|:--:|
| `base_a` | ✅ | ✅ |
| **`ervf`** | ✅ | ✅ |
| `base_b` | ✅ | ✅ |

**G-NERVF-3C geslaagd.** Het volledige model genereert met ERVF overal aan
exact dezelfde 2 × 64 tokens als zonder, en identiek aan het anker dat in de
E4-fase is bevroren. Dat is geen tolerantie maar gelijkheid, en het bevestigt in
de echte lus wat NERVF-2 op het projectievlak mat (0/72 mismatches over vier
breedtes).

## 3. Wat het oplevert

| context | MoE-blok basis | ERVF | speedup | drift | token basis | ERVF | winst | drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 23,711 | 21,060 | 1,126× | 0,199 | 37,660 | **33,959** | **+3,701** | 0,607 |
| 131.072 | 24,104 | 22,371 | 1,078× | 0,453 | 46,780 | **43,678** | **+3,102** | 0,982 |
| 262.100 | 25,550 | 22,330 | 1,144× | 1,506 | 55,640 | **51,135** | **+4,505** | 2,730 |

**Alle drie de tokenwinsten overschrijden hun eigen drift** — conclusief.

Omgerekend binnen deze meting: ctx 0 van 26,55 naar 29,45 tok/s (+10,9%),
262100 van 17,97 naar 19,56 tok/s (+8,8%). De basislijnen hier liggen iets boven
n7b's bevroren cijfers omdat de event-instrumentatie meetelt; alleen het
**verschil** tussen de armen is de meting.

## 4. Waarom de componentpoort faalt, en wat dat wel en niet zegt

**G-NERVF-3P vroeg ≥ 1,35× op de MoE up-projectie en meet 1,144×.** Gefaald, en
niet verruimd.

Maar het venster dat ik geïnstrumenteerd heb omsluit het **hele**
routed-expert-blok van elke MoE-laag: de router, de shared expert, de zes
up-GEMV's, het volledige `down_masked`-pad met scan en gather, en de accumulatie.
ERVF vervangt daarvan alleen de NVFP4-rij-GEMV. Uit S14 en N2 weten we hoe die
27,7 ms verdeeld is: `up` 6,55 · `down_masked` 8,39 · `route` 3,5 · `shared` 3,58
· `host_gap` 4,67 · `accum` 1,0. ERVF raakt `up` en `shared` — samen ~10,1 ms van
de ~25 ms in dit venster.

De poort meet dus een verdunde grootheid. Dat is een tekortkoming van hoe ik hem
geïnstrumenteerd heb ten opzichte van hoe hij geformuleerd was ("MoE
up-projectie"), en ik reken hem als gefaald in plaats van hem opnieuw te
interpreteren. Het onverdunde getal staat in NERVF-2: **1,936× op het
projectievlak, bitexact**.

Een sanity-check die wel klopt: 1,936× op ~10,1 ms voorspelt ~4,9 ms besparing;
gemeten in het MoE-venster is dat 2,65 tot 3,22 ms, en op het token 3,1 tot
4,5 ms. Dezelfde orde, met de overlap van de lus ertussen.

## 5. Op de doorbraakladder

**LEVEL 2 bevestigd** (≥1,35× projectie, exact — NERVF-2: 1,936×).
**LEVEL 3** (≥1,5× volledig expertpad) is niet gehaald: het expertpad bevat de
down-projectie, die ERVF nog niet raakt. Dat is NERVF-4.
LEVEL 4 (≥35 tok/s) is niet gehaald: 29,45 tok/s bij ctx 0 binnen deze meting.

## 6. Claim boundary

In-lus A/B op deze GPU bij capacity 72. CUDA-events omsluiten het hele
routed-expert-blok van elke MoE-laag, dus het MoE-getal is een **component** die
ook router, shared expert en het down-pad bevat; ERVF vervangt daarin alleen de
NVFP4-rij-GEMV, zodat de componentversnelling per constructie verdund is en
**niet** de projectievlak-versnelling van NERVF-2 is. Het tokengetal is
end-to-end wandtijd. Exactheid is een harde poort: de generatie moet identiek
zijn tussen de armen én tegen het bevroren anker. Drie armen begrenzen de drift;
een tokenwinst telt alleen waar zij haar eigen drift overschrijdt. De tok/s-
omrekening in §3 geldt binnen deze meting en is niet vergelijkbaar met n7b's
bevroren baseline. Deze winst mag **niet** worden opgeteld bij die van
attention-v4, graph of gatherless — de combinatieregel eist een nieuwe A/B per
combinatie.

## 7. Artefacten

`scripts/nervf_nemotron/nervf3_integration_ab.json` ·
`nervf3_integration_ab.json` ·
`src/moe_lab/lightningstream_nemotron/fused_nvfp4.py` (`use_ervf`, default uit) ·
`scripts/nervf_nemotron/nervf_independent_verify.py` (66/66, `VERIFIED`) ·
`nervf_independent_verification.json` · `NERVF_SHA256_MANIFEST.json` ·
`reports/lightningstream_nemotron/protected_verification_after_nervf3.json`
(0 modified / 0 removed)
