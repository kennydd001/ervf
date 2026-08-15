# W1 — het hostpad van `_moe_cached` goedkoper maken: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Aanleiding: S14's `host_gap` van 4,7 ms GPU-idle per token, en Y1's meting dat de
per-laag-synchronisatie 6,66 ms waard is bij 262K.

## 1. Wat er gebouwd wordt, en wat uitdrukkelijk niet

**Niet** een device-side router. Die stuit op een echte ontwerpvraag: de sync
bestaat omdat de host bij een **miss** de H2D-kopie moet uitgeven, en missers
rechtstreeks uit mapped host lezen zet PCIe-verkeer op het kritieke pad dat nu op
de copy-stream overlapt. Dat is een aparte fase.

**Wel**: het hostwerk zelf goedkoper maken. S14 mat 4,7 ms GPU-idle per token
terwijl de host de readback afhandelt, de LRU bijwerkt en kopieën uitgeeft. Per
token loopt `_moe_cached` 23 keer, en doet daarbij per expert-call:

- twee tot vier **cupy-slices** (`c["codes"][sl*UP_CODE:(sl+1)*UP_CODE]`), elk een
  nieuw ndarray-object met metadata — ruwweg 400 per token;
- `float(bank["globals"][e, 1])`, een numpy-scalarextractie, 276 per token;
- `bank["down_base_ptr"] + int(e) * DOWN_PANEL_BYTES`, per call opnieuw;
- `self.act[:self.moe_inter]`, nog een slice per call;
- twee numpy-allocaties (`.astype(int)`, `.astype(np.float64)`) per laag;
- `self.cache_stats["hits"] += 1`, een dict-lookup per expert;
- een `with self.copy_stream:`-context per laag, ook als er geen misser is;
- twee list-comprehensions voor de volgorde.

Geen daarvan raakt de GPU. Alle zijn vooraf te berekenen of te vermijden.

## 2. De ingrepen, allemaal semantiek-neutraal

1. Per-slot **views** op de cache, één keer gemaakt in `enable_cache`, in plaats
   van een slice per aanroep.
2. Per-expert **views** op de gepinde bank en een lijst met per-expert
   down-pointers, één keer gemaakt in `load_routed_bank`.
3. De globale schalen als **Python-floats** in een lijst, één keer omgezet.
4. `self.act[:moe_inter]` en `self.act[:shared_inter]` als vaste buffers.
5. De routepakketten met één `tolist()` naar Python in plaats van twee
   numpy-`astype`-allocaties.
6. Hit/miss-telling in lokale ints, één keer per laag teruggeschreven.
7. De copy-stream-context alleen betreden als er werkelijk een misser is.
8. Eén doorloop die slots, wachtvlaggen én volgorde tegelijk opbouwt.

Dezelfde kernels, dezelfde argumenten, dezelfde volgorde. Het pad blijft
`up_only` en `full` beide ondersteunen.

Het nieuwe pad komt naast het bestaande, achter `fast_host` (default **False**),
zodat elke eerdere meting het pad blijft beschrijven dat zij gemeten heeft en de
A/B in één proces kan.

## 3. Poorten

- **G-W1-C1 — semantiek.** De generatie is bit-identiek aan het bestaande pad
  over 2 × 64 tokens. Faalt dit, dan wordt er geen tijd gerapporteerd en gaat de
  wijziging niet in.
- **G-W1-P1 — adoptie.** Winst bij ctx 262100 **≥ 1,0 ms** per token én geen
  regressie bij ctx 0. De drempel is bewust laag ten opzichte van S14's 4,7 ms
  `host_gap`: hij zegt alleen "meer dan de extra code waard", en hij is níét op
  een gemeten uitkomst gekalibreerd, want die bestaat nog niet.
- **G-W1-D1 — drift.** Gebracketeerde basislijnen; de winst telt alleen als zij
  groter is dan `|base₂ − base₁|`.
- **G-W1-S1 — bovengrens.** De winst mag S14's `host_gap` (4,7 ms bij 262K) niet
  overschrijden. Doet zij dat wel, dan meet de opzet iets anders dan hostwerk.

Poorten worden na het zien van het resultaat niet verruimd.

## 4. Wat dit niet is

Geen device-side routing, geen wijziging aan de kernels, geen wijziging aan de
routebeslissing, geen quantisatie, geen speculatie. De winst is begrensd door
`host_gap` en kan de plafonds uit Z1 en Y2-R1 niet verzetten.

## 5. Artefacten

`src/moe_lab/lightningstream_nemotron/runtime.py` (`fast_host`, default uit) ·
`scripts/lightningstream_nemotron/w1_host_path_ab.py` · `w1_host_path_ab.json` ·
`scripts/lightningstream_nemotron/w1_independent_verify.py` ·
`w1_independent_verification.json` · rapport met claim boundary.

## 6. Claim boundary van dit document

Geen meting, geen resultaat.
