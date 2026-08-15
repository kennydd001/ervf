# S11 — volledig-record caching vs. up-only: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.** Geen meting van deze fase bestond toen dit
document geschreven werd.
Model: `models/nemotron_3_5_lightning_v35`, `LS_MODEL_DIR=nemotron_3_5_lightning_v35`
Aanleiding: S9's onopgeloste punt, letterlijk overgenomen uit de overdracht.

## 1. De vraag, met één variabele

Sinds S5 is de cache **up-only**. Een cacheslot bevat `up_proj`-codes en
-scales (2.806.272 B) en verder niets. `down_proj` wordt bij **elke** expertcall
opnieuw uit mapped host-geheugen gehaald door `gather_down_sparse` (gemeten op
25,05 GB/s, tegen ~250 GB/s device-geheugen), ook als de expert in de cache zit.
Een hit bespaart dus alleen de up-helft.

De vraag is niet "helpt cachen" maar: **wat kun je het beste met dezelfde bytes
doen?**

| arm | wat er in een slot zit | slotgrootte | capacity | cachebytes |
|---|---|---:|---:|---:|
| **A (huidig)** | up-codes + up-scales | 2.806.272 B | 72 | 4,328 GiB |
| **B (nieuw)** | idem **plus** het panel-major `down`-record | 5.612.544 B | 36 | 4,328 GiB |

Exact 2× de slotgrootte, exact de halve capacity: **de cache is in beide armen
byte-voor-byte even groot.** De enige variabele is wat erin ligt.

## 2. Wat er verandert in de dataplane

Arm B raakt twee plekken:

- `enable_cache(capacity, mode)` alloceert in mode `full` een extra
  `down`-buffer van `capacity × 2.806.272 B` per laag.
- `_moe_cached` kopieert bij een **miss** ook het `down`-record naar het slot
  (dezelfde copy-stream, hetzelfde event), en roept bij een **hit** de bestaande
  `down_masked_into(..., gather_from_host=False)` aan met een **device**-pointer
  naar het slot in plaats van een host-pointer.

Die tweede stap bestaat al: `down_masked_into` heeft `gather_from_host` al als
parameter en de masked GEMV leest in beide gevallen uit device-geheugen, alleen
in arm A uit de sparse `mirror` en in arm B uit het volledige slot. De
ontvlechting die de overdracht vroeg is daarmee één vlag, geen herbouw.

`mode="up_only"` blijft de default. Arm A is bit-voor-bit het pad dat n7b nu
draait.

## 3. Wat elke arm kost, vooraf uitgerekend (geen voorspelling)

Per token doet de runtime 138 expertcalls (23 lagen × 6).

| | arm A @72 | arm B @36 |
|---|---:|---:|
| gemeten hitrate | 0,804 (n7b) | onbekend; N7-A simuleerde 0,650 bij capacity 32 |
| misses per token | ≈ 27 | ≈ 46 bij hitrate 0,67 |
| PCIe per miss | 2,81 MB (alleen up) | 5,61 MB (up + down) |
| PCIe uit misses | ≈ 76 MB | ≈ 258 MB |
| `gather_down_sparse`-calls | **138** (ook op hits) | alleen op misses ≈ 46 |
| PCIe uit gathers (~9% van down) | ≈ 35 MB | ≈ 0 |

Arm B verplaatst dus werk van de **compute-stream** (de gather loopt synchroon
in het kritieke pad van elke expert) naar de **copy-stream** (missetransfers,
die S8's negatieve restpost van −16,9 ms laat zien dat ze grotendeels overlappen).
Het kost meer bytes en minder kritiek pad. Welke kant zwaarder weegt is precies
wat niet uit te rekenen valt, en daarom wordt het gemeten en niet beredeneerd.

## 4. Meetopzet

Eén proces, één modelload, drie armen in deze volgorde:

1. **A₁** — up-only, capacity 72
2. **B** — full-record, capacity 36
3. **A₂** — up-only, capacity 72, opnieuw

A₂ is er om drift te begrenzen: alles wat tussen A₁ en A₂ verschilt is ruis en
niet het effect van arm B. Als |A₂ − A₁| bij 262100 groter is dan het verschil
tussen B en A₁, is de meting niet-conclusief en wordt er niets aangenomen.

Contexten: **0, 131.072, 262.100**, met dezelfde warm-up- en samplemethode als
`n7b_cached_decode.py` (32 opwarmstappen op die diepte, dan 16 gemeten stappen,
p50), en dezelfde gevarieerde token-ids, zodat de router per stap echt van route
wisselt.

Configuratie verder identiek aan de baseline: `--embed-on-host`, FP8-KV,
`max_ctx 262144`.

## 5. Poorten

- **G-S11-C1 — correctheid.** De generatie van arm B is **bit-identiek** aan die
  van arm A over 2 × 64 tokens. Beide armen lezen dezelfde checkpointbytes; alleen
  de residentie verschilt. Een verschil is dus een fout, geen afweging. Faalt
  deze poort, dan worden er géén prestatiecijfers van arm B gerapporteerd.
- **G-S11-P1 — adoptie.** Arm B wordt alleen aangenomen als hij bij ctx 262100
  **≥ 3% hogere p50-tok/s** haalt dan A₁ **én** bij ctx 0 niet onder A₁ zakt.
  De 3% is gekozen omdat de reproductie van vandaag armen tot 2,6% zag
  variëren tussen twee identieke runs; alles daaronder is niet van ruis te
  onderscheiden.
- **G-S11-D1 — drift.** |A₂ − A₁| bij 262100 moet kleiner zijn dan |B − A₁|,
  anders is de uitkomst niet-conclusief ongeacht het teken.

Deze poorten worden na het zien van het resultaat niet verruimd, niet verlaagd
en niet per context heronderhandeld. Als arm B faalt, blijft up-only staan en
wordt dat als weerlegde hypothese vastgelegd — niet als "veelbelovend maar
ongetuned".

## 6. Wat deze fase níét doet

- Geen andere capacity dan 72 / 36. Een capacity-sweep zou een tweede variabele
  introduceren.
- Geen wijziging aan de masked GEMV, de gather-kernel, de router of de shared
  expert.
- Geen uitspraak over waar de resterende ~30 ms van de MoE-term zit (S9). Deze
  meting kan die vraag beantwoorden noch beantwoorden-door-aftrekking.
- Geen componentmeting die naar tok/s wordt opgewaardeerd; de armen worden
  end-to-end vergeleken, wat juist de reden is dat S8's overtel-probleem hier
  niet speelt.

## 7. Artefacten

`src/moe_lab/lightningstream_nemotron/runtime.py` (mode-parameter, default
ongewijzigd) · `scripts/lightningstream_nemotron/s11_cache_mode_ab.py` ·
`s11_cache_mode_ab.json` ·
`scripts/lightningstream_nemotron/s11_independent_verify.py` ·
`s11_independent_verification.json` · `protected_verification_after_s11.json` ·
rapport met claim boundary.

## 8. Claim boundary van dit document

Geen meting, geen resultaat. De tabel in §3 is aritmetiek op eerder gemeten
byte- en hitrate-getallen, expliciet geen voorspelling.
