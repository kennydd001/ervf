# S14 — GPU-event-tijdslijn van de MoE-laag: de restpost heeft een naam

Datum: 2026-08-15
Verdict: **Alle poorten gehaald. De MoE-lagen bezetten in de echte lus 27,7 ms stream-wandtijd per token (beide contexten), niet de 39,5 ms die S8's geïsoleerde meting toekende. Het verschil was nooit MoE-werk: het is werk van attention en Mamba dat in de wachtrij staat en pas op de host-klok landt waar de lus zijn enige sync heeft — de route-readback. Binnen de MoE-laag is nu elke microseconde toegeschreven; de grootste niet-rekenende post is 4,7 ms GPU-stilstand per token terwijl de host readback-afhandeling, LRU-boekhouding en copy-issue doet.**
Terminal state: `s14_moe_timeline_attributed_residual_named`
Preregistratie: `S14_MOE_LAYER_TIMELINE_PREREGISTRATION_2026-08-15.md` (bevroren vóór uitvoering)

## 1. Methode in één alinea

S8 toonde aan dat geïsoleerd timen hier 16,9 ms te veel telt omdat de sync de
overlap vernietigt; S12 mat daarom marginalen in de lus maar kon per constructie
niet zien *waar* de rest zat. Deze fase zet **CUDA-events met timing** tussen de
fasen van `_moe_cached` — een timestamp op een stream, geen synchronisatie — via
een subklasse in het runnerscript. `runtime.py` is onaangeraakt (input lock +
verifier bevestigen de hash). Elke microseconde van elke MoE-laag valt in
precies één segment, omdat segmenten opeenvolgende event-paren op dezelfde
stream zijn. Armen `base0 · probed · base1` (probelast-poort), contexten 0 en
262.100 (pos-sprongprotocol zoals N7-B/S12), capacity 72, 16 tokens per context.

Poorten: G-S14-C1 (generatie bit-identiek, 2×32 tokens) ✅ · G-S14-P1 (probelast
+4,97 ms bij ctx 0 resp. +3,00 ms bij 262K boven de gebracketeerde basislijn,
beide < 20%) ✅ conclusief · G-S14-S1 (segmentsom 27,7 ≤ token-p50, ≥ de helft
van de S8-term bij 262K, geen negatieve segmenten; de verifier bevestigt bovendien
dat de deelsegmenten per token optellen tot `layer_total` binnen 0,05 ms) ✅.

## 2. De tijdslijn (gemiddelde ms per token, 23 MoE-lagen, 16 tokens)

| segment | ctx 0 | ctx 262.100 | wat het is |
|---|---:|---:|---|
| `route` | 3,583 | 3,503 | router-GEMV + sigmoid + argsort + pack |
| `shared_up` + `shared_dn` | 3,445 | 3,576 | shared expert (de bestaande overlap) |
| `host_gap` | 5,058 | 4,672 | **stream stil** terwijl host readback-afhandeling, LRU en copy-issue doet |
| `up` (138 experts) | 6,715 | 6,551 | up-GEMV's, 47 µs/expert |
| `down_masked` | 7,830 | 8,393 | sparse host-gather + masked GEMV, 61 µs/expert |
| `accum` | 1,111 | 0,995 | gewogen optelling |
| **`layer_total`** | **27,741** | **27,690** | stream-wandtijd van de MoE-lagen |
| (memo) `miss_copy_batch` | 2,817 | 2,325 | copy-stream, loopt naast de compute-stream |
| (memo) `readback_host` | 12,968 | 31,010 | host-wandtijd van de readback, zie §3 |
| (memo) experts met wait | 18,9/138 | 15,6/138 | misses die op hun eigen event wachten |

De MoE-lagen zijn **context-onafhankelijk** (27,7 ms bij beide diepten), zoals
het ontwerp van de dataplane ook bedoelt. De token-p50 loopt van 41,2 naar 57,2 ms
(probed); dat verschil zit volledig buiten de MoE-lagen.

## 3. De restpost was een meetartefact plus drie echte termen

S8 kende 39,523 ms aan "MoE" toe bij 262K. De tijdslijn laat zien wat dat getal
werkelijk bevatte:

- **27,7 ms** echte MoE-streamtijd (deze fase);
- **~12 ms** werk van ándere lagen dat bij de geïsoleerde meting in de MoE-term
  terechtkwam. Direct bewijs uit deze run: `readback_host` — de enige sync van
  de lus, waar alles wat nog in de wachtrij staat op de host-klok landt — is
  bij 262K **18,0 ms per token groter** dan bij ctx 0 (31,010 − 12,968), terwijl
  alle MoE-segmenten context-onafhankelijk zijn. Dat verschil is de in de
  wachtrij staande attention (S8: 18,634 ms bij 262K), die bij elke van de 23
  readbacks deels afdraait. Dezelfde grootheid, twee onafhankelijke metingen.

Wat S12 "geen naam" gaf, blijkt dus: geen onbekende MoE-kost, maar (a) het
afdrain-artefact van de geïsoleerde meting en (b) drie echte, nu benoemde
segmenten die de marginale methode niet kon zien: `host_gap` (4,7 ms GPU-
stilstand), `route` (3,5 ms voor 344 kFLOP aan arithmetiek — puur launch-
gebonden: een tiental kernel-launches per laag) en de in-lus vertraging van
`up`/`down` ten opzichte van S9's microbenchmark (6,6+8,4 = 15,0 ms in de lus
tegen ~9,0 ms geïsoleerd; koude L2 en de PCIe-gather in `down_masked`).

## 4. Wat dit betekent voor de 50-tok/s-vraag (rekenwerk op gemeten vloeren)

Geen meting, aritmetiek op de gemeten segmenten en eerdere vloeren, bij 262K:

| term | nu (ms/token) | gemeten vloer | bron van de vloer |
|---|---:|---:|---|
| MoE-lagen | 27,7 | ~16 (expert-GEMV's + gather; router-fusie en host_gap-wegwerken zijn bekende technieken, geen vloer) | S9 + deze fase |
| attention (6 lagen) | ~18,6 | 3,3 (KV eenmaal lezen: 0,548 ms/laag bij 244,8 GB/s, S7-mechanismecheck) | S7 |
| Mamba (23 lagen) | ~8,3 | onbekend (alleen geïsoleerd gemeten) | S8, met overtel-voorbehoud |
| `lm_head` | ~2,1 | ~2,1 | S8 |

Zelfs met álles op de gemeten of geraamde vloer: 16 + 3,3 + 8,3 + 2,1 ≈ **30 ms
per token ≈ 33 tok/s bij 262K**. De helft daarvan is KV-lezen en expert-bytes —
fysieke minimaalverplaatsing bij ongewijzigde semantiek. 50 tok/s bij 262K
vereist 20 ms en ligt daarmee **buiten de gemeten fysica van dit model op deze
GPU**, ongeacht verdere kernel- of dataplanningswerk. Dat is het antwoord dat de
eerdere fases stuk voor stuk dichterbij brachten; S14 maakt het expliciet
aantoonbaar. (Bij kortere context verschuift de grens: bij ctx 0 is attention
~0 en staat de MoE-term van 27,7 ms alleen al voor ~36 tok/s als alles else
gratis zou zijn.)

## 5. Wat deze fase niet doet

Geen optimalisatie gebouwd, geen tok/s-verbetering gemeten, geen kwaliteitsclaim.
`miss_copy_batch` is op de copy-stream gemeten en telt bewust niet mee in
`layer_total`. `readback_host` is een host-wandtijd die grotendeels uit wachten
op legitiem in de wachtrij staand werk bestaat — hij is geen componentkost en
wordt nergens als zodanig opgeteld. De 262K-arm gebruikt het pos-sprongprotocol
(64 echte tokens, `pos` gezet, 32 warm, 16 gemeten): de KV-inhoud is
deels synthetisch, de timingstructuur is dat niet — hetzelfde protocol als N7-B
en S12.

## 6. Onafhankelijke verificatie

`s14_independent_verify.py` importeert niets uit de runner, herberekent alle
segmentgemiddelden en p50's uit de ruwe per-token data, verifieert per token dat
de deelsegmenten optellen tot `layer_total` (≤ 0,05 ms afwijking, 32/32 tokens),
her-evalueert alle vijf de poorten en controleert de input-lock-hashes van
runner en `runtime.py`. **Alle checks geslaagd, verdict `VERIFIED`.**

Protected manifest na deze fase: zie `protected_verification_after_s14.json`.

## 7. Claim boundary

Een CUDA-event-tijdslijn van de echte decode-lus op deze GPU, capacity 72,
contexten 0 en 262.100 (pos-sprong), 16 tokens per context. Stream-segmenten
zijn wandtijd op de stream en bevatten werk én event-wachten naar ontwerp. Niets
hiervan is naar tokens per seconde omgerekend behalve de expliciet als
rekenwerk gelabelde vloer-som in §4. Geen uitspraak over andere hardware,
batchgroottes of prompts.

## 8. Artefacten

`S14_MOE_LAYER_TIMELINE_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/s14_moe_layer_timeline.py` ·
`s14_moe_layer_timeline.json` ·
`scripts/lightningstream_nemotron/s14_independent_verify.py` ·
`s14_independent_verification.json` · `s14_input_lock.json` ·
`protected_verification_after_s14.json`
