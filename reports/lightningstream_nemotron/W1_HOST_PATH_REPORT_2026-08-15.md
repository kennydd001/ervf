# W1 — het hostpad goedkoper: gebouwd, bit-identiek, +5,5% bij ctx 0, poort gefaald

Datum: 2026-08-15
Verdict: **Gebouwd en exact: de generatie is bit-identiek in alle drie de armen. Bij ctx 0 is de winst conclusief +2,206 ms (+5,5%, drift 0,047). Bij 128K en 262K is zij ~0,5 ms en verdwijnt zij in een drift van 2,6 respectievelijk 4,5 ms. G-W1-P1 vroeg ≥ 1,0 ms bij 262100 en meet +0,511. Poort gefaald; `fast_host` blijft opt-in en de default verandert niet.**
Terminal state: `w1_host_path_exact_gain_at_ctx0_gate_failed_at_depth`
Preregistratie: `W1_HOST_PATH_PREREGISTRATION_2026-08-15.md`

## 1. Wat er gebouwd is

Niet de device-side router — die stuit op een echte ontwerpvraag (de sync bestaat
omdat de host bij een miss de H2D-kopie uitgeeft; missers uit mapped host lezen
verplaatst PCIe-verkeer naar het kritieke pad). Wel het hostwerk eromheen.

`_moe_cached_fast` staat náást `_moe_cached`, achter `fast_host` (default
**False**), met dezelfde kernels, argumenten en volgorde. Weggehaald is alleen
Python:

| was | is nu |
|---|---|
| ~400 cupy-slices per token op cache en `act` | views, één keer gemaakt in `enable_cache` en `_alloc_state` |
| 276 × `float(bank["globals"][e, k])` | twee lijsten Python-floats, één keer omgezet |
| per call `down_base_ptr + e * DOWN_PANEL_BYTES` | een lijst per-expert-pointers |
| per laag twee numpy-`astype`-allocaties | één `tolist()` |
| 138 × `cache_stats[...] += 1` | lokale ints, één keer per laag teruggeschreven |
| `with copy_stream:` per laag, altijd | alleen als er werkelijk een misser is |
| twee list-comprehensions voor de volgorde | één doorloop die slots, wachtvlaggen én volgorde bouwt |

Ook de gepinde bank krijgt per-expert numpy-views, zodat de misser-kopie geen
slice meer bouwt.

## 2. Wat het doet

Drie armen, één proces, één modelload, 32 samples per context per arm.

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-W1-C1** | generatie bit-identiek, 2 × 64 tokens | identiek in alle drie de armen | ✅ |
| **G-W1-P1** | ≥ 1,0 ms winst bij 262100, geen regressie bij ctx 0 | **+0,511 ms** | ❌ |
| **G-W1-S1** | winst ≤ S14's `host_gap` | overal binnen | ✅ |

| context | basislijn | fast | winst | lokale drift | conclusief |
|---:|---:|---:|---:|---:|:--:|
| 0 | 39,877 ms | 37,671 ms | **+2,206 ms (+5,5%)** | 0,047 | ✅ |
| 131.072 | 49,144 | 48,706 | +0,439 (+0,9%) | 2,591 | ❌ |
| 262.100 | 58,384 | 57,873 | +0,511 (+0,9%) | 4,520 | ❌ |

Verifier 39/39, `VERIFIED`. Protected 0 modified / 0 removed.

## 3. Wat dit zegt

**Bij ctx 0 is het echt en scherp gemeten**: 2,206 ms winst tegen 0,047 ms drift
is een verhouding van 47:1. Dat is 44% van S14's `host_gap` van 5,058 ms bij die
diepte — het Python-werk was dus ongeveer de helft van die leegloop, en de rest
zit in de readback-wachttijd zelf, die deze ingreep niet raakt.

**Bij diepe context verdwijnt het.** Dat is geen tegenspraak maar precies wat je
verwacht: bij 262K is het token 58 ms in plaats van 40, en de extra 18 ms is
attention-werk waarachter de host zich kan verstoppen. Hostkosten tellen alleen
mee zolang de GPU niets te doen heeft.

De poort vroeg 1,0 ms bij 262100 omdat dat de diepte is waar de gebruiker naar
kijkt. Die haalt hij niet, en hij wordt niet verlaagd. Daarom blijft `fast_host`
opt-in en verandert de default niet.

**De drift bij diepe context is het echte obstakel voor de meting**, niet voor de
ingreep: 2,6 en 4,5 ms tussen twee identieke armen, tegen 0,047 ms bij ctx 0. De
GPU draait op 86–87 °C (S14) en de armen zijn lang. Een opzet met kortere,
vaker afgewisselde armen zou de 0,5 ms kunnen oplossen — maar dat is een nieuwe
preregistratie, en het resultaat van déze staat.

## 4. Wat het niet verandert

De plafonds blijven staan. Z1 zette de bovengrens van elke boomverifier op
45–61 tok/s, Y2-R1 begrenst de byte-tak op 34% van de GEMV, en X1 sloot
speculatie. Een winst die door `host_gap` begrensd is, kan daar niets aan
verzetten — en dat was vooraf ook zo opgeschreven.

Toegepast op de bevroren basislijn zou de ctx-0-winst 36,05 → 34,1 ms zijn. Bij
262K is er binnen deze meting geen winst aan te tonen.

## 5. Claim boundary

Gemeten batch-1 single-stream decode op deze GPU, drie armen in één proces tegen
één modelload, met hetzelfde warm-up- en sampleprotocol als de bestaande runner.
De enige variabele is of het per-expert-hostwerk vooraf berekend is; de kernels,
hun argumenten en hun volgorde zijn ongewijzigd, en dat is wat de
identiteitspoort controleert. De winst bij ctx 0 is conclusief; die bij 128K en
262K niet, en wordt daarom niet als winst geclaimd. Geen kwaliteitsclaim, geen
uitspraak over andere hardware of capacities.

## 6. Artefacten

`W1_HOST_PATH_PREREGISTRATION_2026-08-15.md` ·
`src/moe_lab/lightningstream_nemotron/runtime.py` (`fast_host`, default uit) ·
`scripts/lightningstream_nemotron/w1_host_path_ab.py` · `w1_host_path_ab.json` ·
`scripts/lightningstream_nemotron/w1_independent_verify.py` ·
`w1_independent_verification.json` · `protected_verification_after_w1.json`
