# S11 — volledig-record caching: weerlegd bij gelijke bytes

Datum: 2026-08-15
Verdict: **Arm B verliest op alle drie de contextdiepten. G-S11-P1 gefaald met −4,84% bij 262100 tegen een poort van +3%. Up-only blijft staan. De uitkomst weerlegt ook de premisse achter de hypothese: 2,9× meer PCIe-verkeer kost maar 4,8%, dus de MoE-term is niet transfergebonden.**
Terminal state: `s11_full_record_cache_refuted_moe_not_transfer_bound`
Preregistratie: `S11_FULL_RECORD_CACHE_PREREGISTRATION_2026-08-15.md` (bevroren vóór uitvoering)

## 1. De meting

Drie armen, één proces, één modelload, identiek warm-up- en sampleprotocol aan
`n7b_cached_decode.py`. Beide cachemodi kregen **exact evenveel bytes**: een
volledig-record-slot is precies twee keer een up-only-slot, dus capacity 36 en
capacity 72 alloceren allebei 4,328 GiB. Onafhankelijk geverifieerd tegen de
safetensors-headers van het checkpoint: up-helft 2.806.272 B, down-helft
2.806.272 B — exact gelijk.

| context | A₁ up-only @72 | **B full @36** | A₂ up-only @72 | effect B−A₁ | drift A₂−A₁ |
|---:|---:|---:|---:|---:|---:|
| 0 | 27,078 | **25,551** | 26,391 | −5,64% | −2,54% |
| 131.072 | 21,466 | **20,903** | 21,590 | −2,62% | +0,58% |
| 262.100 | 18,227 | **17,346** | 18,185 | −4,84% | −0,23% |

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-S11-C1** | generatie bit-identiek, 2 × 64 tokens | identiek in alle drie de armen | ✅ |
| **G-S11-P1** | ≥ +3% bij 262100 én geen regressie bij ctx 0 | **−4,84%**, ook ctx 0 lager | ❌ |
| **G-S11-D1** | \|A₂−A₁\| < \|B−A₁\| bij 262100 | 0,042 vs 0,881 (21×) | ✅ conclusief |

De correctheidspoort is niet triviaal gehaald: arm B leest `down` uit een
residente kopie in plaats van uit de sparse host-gather, en de generatie is
token-voor-token identiek over beide prompts. De dataplane-wissel is dus exact,
en het prestatieverschil is een echt prestatieverschil en geen semantiekverschil.

De herhaling A₂ is de reden dat dit een conclusie mag heten: bij 262100 wijkt A₂
0,23% van A₁ af, tegen een effect van 4,84%. Hits en misses waren in A₁ en A₂
tot op het getal identiek (4.530 misses in beide), dus de meetlus zelf is
deterministisch.

## 2. Waarom arm B verliest

| | A₁ up-only @72 | B full @36 |
|---|---:|---:|
| expert-lookups in de sweep | 37.531 | 37.531 |
| hitrate | **0,8793** | **0,6901** |
| misses | 4.530 | **11.631** |
| bytes per miss | 2.806.272 | 5.612.544 |
| miss-verkeer over PCIe | ≈ 12,7 GB | ≈ 65,3 GB |
| `gather_down_sparse`-calls | 37.531 | **0** |
| gather-verkeer (~9% van down, S5) | ≈ 9,5 GB | 0 |
| **totaal PCIe** | **≈ 22,2 GB** | **≈ 65,3 GB** |

Halvering van de capacity kost 0,879 → 0,690 hitrate, dus **2,57× zoveel
misses**, en elke miss verplaatst het dubbele. Dat is 5,1× het miss-verkeer. De
besparing — alle 37.531 gathers vervallen, want `down` is nu resident — weegt
daar niet tegenop.

## 3. Wat dit los van de poort oplevert

Arm B verplaatst **2,9× zoveel bytes over PCIe** en verliest daarmee **4,8%**.
Was de MoE-term transfergebonden geweest, dan had bijna een verdrievoudiging van
het PCIe-verkeer catastrofaal moeten uitpakken. Dat gebeurt niet.

Dat is een tweede, onafhankelijke bevestiging van S8: de premisse waarop S5 werd
gebouwd — "miss-bytes over PCIe domineren" — geldt niet meer voor deze runtime.
En het maakt de resterende vraag scherper in plaats van breder: van de 39,5 ms
MoE-term verklaren de GEMV's ~9,0 ms (S9) en de transfer aantoonbaar weinig. De
~30 ms die overblijft zit dus in het pad zelf, niet in wat er over de bus gaat.

Meteen ook een streep door een aantrekkelijk klinkende vervolgstap: "cache
`down` er ook bij zodra er VRAM vrijkomt" is nu gemeten en verworpen. VRAM aan
`down` besteden is minder waard dan hetzelfde VRAM aan méér `up`-slots.

## 4. Wat deze fase niet doet

- Geen capacity-sweep. Dat zou een tweede variabele zijn en de vraag was
  expliciet "wat kun je het beste met dezelfde bytes doen".
- Geen scheiding tussen "de hitrate zakt" en "de misses zijn duurder". Die twee
  zijn bij gelijke bytes onlosmakelijk; wie ze wil scheiden moet up-only @36
  meten, en dat is een aparte vraag met een aparte preregistratie. Voor de
  adoptiebeslissing is het niet nodig.
- Geen verklaring van de ~30 ms restpost in de MoE-term. Deze meting begrenst
  hem, en zeker niet door aftrekking.

## 5. Onafhankelijke verificatie

`s11_independent_verify.py` importeert niets uit de runner of de runtime. Het
leest de slotgroottes rechtstreeks uit de safetensors-headers van het checkpoint
in plaats van uit constanten in `runtime.py`, herberekent elke p50 uit de ruwe
milliseconde-samples met een eigen mediaan, herhaalt de identiteitsvergelijking
op de token-ids en evalueert de drie poorten opnieuw. **41 van 41 checks,
verdict `VERIFIED`.**

Protected manifest na deze fase: **0 modified / 0 removed**.

## 6. Wat er in de code veranderd is

`enable_cache(capacity, mode)` kent nu `up_only` (default, ongewijzigd gedrag) en
`full`. In `full` krijgt elk slot ook het panel-major `down`-record, en gebruikt
`_moe_cached` de al bestaande parameter `down_masked_into(..., gather_from_host=False)`
met een device-pointer. De ontvlechting die de overdracht vroeg bleek daarmee één
vlag te zijn: `down_masked_into` was al geparameteriseerd op de herkomst van de
bytes. De default blijft `up_only`, dus elke eerdere meting beschrijft nog steeds
het pad dat zij gemeten heeft.

## 7. Claim boundary

Gemeten batch-1 single-stream decode op deze GPU, drie armen in één proces tegen
één modelload, met hetzelfde warm-up- en sampleprotocol als de bestaande runner.
De twee cachemodi houden exact evenveel bytes vast. Geen kwaliteitsclaim, geen
benchmarkscore, geen uitspraak over andere hardware, batchgroottes of capacities.
De PCIe-getallen in §2 zijn aritmetiek op gemeten miss-tellingen en bekende
recordgroottes, geen busmeting; het percentage van de gather-bytes komt uit S5.

## 8. Artefacten

`S11_FULL_RECORD_CACHE_PREREGISTRATION_2026-08-15.md` ·
`src/moe_lab/lightningstream_nemotron/runtime.py` (mode-parameter) ·
`scripts/lightningstream_nemotron/s11_cache_mode_ab.py` · `s11_cache_mode_ab.json` ·
`scripts/lightningstream_nemotron/s11_independent_verify.py` ·
`s11_independent_verification.json` · `protected_verification_after_s11.json`
