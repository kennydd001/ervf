# Y1/Y2 — de readback is 6,7 ms waard, en bytes snijden koopt 34% van de GEMV

Datum: 2026-08-15
Verdict: **Y1 gehaald en groot: het wegnemen van de per-laag device→host-sync levert +6,656 ms per token bij 262K en +4,359 ms bij ctx 0, bit-identiek. Y2 gefaald maar nipt: de bytes halveren bespaart 34,2% van de GEMV, niet de 40% die de poort vroeg — de kernel is voor 31,6% byte-onafhankelijk.**
Terminal state: `y1_readback_worth_6_7ms_y2_bytes_bounded_at_34pct`
Preregistratie: `Y1Y2_READBACK_AND_BYTES_PREREGISTRATION_2026-08-15.md`

## 1. Y1 — wat de host-round-trip kost

De routed lus synchroniseert per MoE-laag omdat de expert-ids naar de host moeten
om de gepinde bank te indexeren: 23 syncs per token. Kimi's S14 mat daar met
CUDA-events `host_gap` 4,7 ms GPU-idle plus een launch-gebonden `route` van
3,5 ms omheen.

Arm B draait de échte lus, laat `_route_device` volledig op device draaien —
GEMV, sigmoid, bias, argsort, werk dat in elke implementatie blijft bestaan —
maar **leest hem niet terug**: de ids komen uit een capture van exact dezelfde
run. Per constructie identieke routes.

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-Y1-C1** | generatie bit-identiek, 2 × 64 tokens | identiek | ✅ |

| context | basislijn | zonder readback | winst | lokale drift |
|---:|---:|---:|---:|---:|
| 0 | 38,852 ms | 34,493 ms | **+4,359 ms (11,2%)** | 2,121 |
| 262.100 | 58,690 ms | 52,034 ms | **+6,656 ms (11,3%)** | 0,919 |

Beide conclusief (winst > drift) en beide binnen S14's grens van
`host_gap + route` (8,641 / 8,175 ms), wat G-Y1-S1 eist: de opzet meet de sync en
niet iets anders. Verifier 33/33.

**Dit is de grootste niet-semantische winst die in dit model gevonden is.**
Toegepast op de bevroren basislijn: 54,28 → 47,6 ms bij 262K, 36,05 → 31,7 ms bij
ctx 0. Dat is aritmetiek op een gemeten winst, geen doorvoermeting — maar de
winst zelf is end-to-end gemeten en bit-identiek.

Wat arm B **niet** is: een implementatie. De LRU-bookkeeping en het uitgeven van
de kopieën blijven host-werk; zonder sync overlappen ze met GPU-werk in plaats
van te stallen. Een echte device-side router moet dat ook oplossen, en zal er
device-side slot-tabellen voor nodig hebben. Wat hier vaststaat is de **prijs van
de sync**, niet dat een gebouwde versie precies dit oplevert.

## 2. Y2 — koopt het snijden van bytes tijd?

ExactFlow A (OrbitANS), B (PathQ), C (CertiPlane) en D (Nested-QAD) delen één
premisse: minder bytes per gewicht → minder tijd. Y2 varieert de recordgrootte
van een écht `up_proj` met de structuur ongewijzigd.

**De eerste pass was fout gemeten** en dat was zichtbaar in het resultaat: hij
synchroniseerde na élke kernel-call, waardoor een vaste ~7 µs launch plus sync
bovenop een ~40 µs kernel kwam. De curve werd niet-monotoon — 75% trager dan
100%, 25% trager dan 50% — de handtekening van een dominante vaste kost. Opnieuw
gemeten met 200 calls per sync, zoals S9's blokgrootte-probe:

| aandeel | kolommen | bytes | µs/call | GB/s |
|---:|---:|---:|---:|---:|
| 100,0% | 2688 | 2.806.272 | 34,472 | 81,4 |
| 87,5% | 2352 | 2.455.488 | 30,822 | 79,7 |
| 75,0% | 2016 | 2.104.704 | 26,703 | 78,8 |
| 50,0% | 1344 | 1.403.136 | 22,692 | 61,8 |
| 25,0% | 672 | 701.568 | 16,781 | 41,8 |

Monotoon, zoals het hoort. Lineaire fit: **8,13 µs per MB plus 10,90 µs vast**,
en die vaste term is **31,6%** van een volledige call.

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-Y2-1** | halvering bespaart ≥ 40% | **34,2%** | ❌ |

De poort faalt, maar nipt, en de betekenis is genuanceerder dan "dood": de kernel
is voor ongeveer twee derde byte-gebonden en voor een derde niet. Een codec die
de expert-bytes halveert wint 34% van de GEMV, geen 50%. Effectieve bandbreedte
81,4 GB/s tegen een device-roofline van ~250 GB/s — nog altijd 3× eraf,
consistent met S9's 86,5 GB/s.

## 3. Wat de twee samen betekenen

Op S14's gemeten segmenten bij 262K (`route` 3,503 · `shared` 3,576 ·
`host_gap` 4,672 · `up` 6,551 · `down_masked` 8,393 · `accum` 0,995):

| ingreep | onderbouwing | winst |
|---|---|---:|
| readback weg | Y1, gemeten end-to-end | 6,66 ms |
| expert-bytes halveren | Y2-R1 34,2% × (`up` + `down`) | 5,11 ms |
| shared expert idem | Y2-R1 34,2% × `shared` | 1,22 ms |
| **samen** | | **12,99 ms** |

54,28 − 12,99 = **41,3 ms per token bij 262K**. Bij ctx 0: 36,05 − 4,36 − 5,11 −
1,22 = **25,4 ms**.

Dit is **aritmetiek op gemeten componenten**, geen meting en geen voorspelling.
Het is bovendien de gunstigste lezing: het veronderstelt dat een 2-bits codec
bestaat, zijn kwaliteitspoorten haalt, en dat een device-side router de hele
sync-winst realiseert.

En dan nog: 25,4 ms bij ctx 0 en 41,3 ms bij 262K. De 20 ms die 50 tok/s vraagt
zit er niet in. Bij ctx 0 scheelt het een factor 1,27; bij 262K een factor 2,07.

## 4. Wat er nu van beide packs getest is

| hypothese | status |
|---|---|
| LightningSpec H2/H3 · ExactFlow E (SweepSpec) | **gebouwd en weerlegd** (X1: ratio 1,0017 tegen poort 0,6228) |
| LightningSpec P0 / S13 (route-unie) | **gemeten, poort gefaald** (19,88 resp. 19,512 tegen pariteit 18,683) |
| LightningSpec H1 punt 4 (MicroSpec-vocabulaire) | **gemeten**, keten −44,9%, recall 0,932 < 0,995 |
| LightningSpec H1 punten 1–3 (kwantisatie/windowing van de draft) | **begrensd** door K1: experts 17%, attention 6% van de keten |
| LightningSpec H4/H6/H7/H8 (bredere bomen, andere drafters) | **begrensd door X1**: kosten volgen `B`, dus zelfs perfecte acceptatie breekt hooguit quitte |
| ExactFlow A (OrbitANS) | **begrensd** door Y2-R1: 12–20% packreductie → 4–7% van de GEMV |
| ExactFlow B (PathQ) · C (CertiPlane) · D (Nested-QAD) | **begrensd** door Y2-R1: halvering → 34% van de GEMV |
| ExactFlow F (ElasticDraft) | **begrensd door X1**, want het is een draftverbetering |
| **device-side routing** (niet in de packs) | **gemeten: 6,66 ms bij 262K** — de grootste vondst |

Er staat geen hypothese meer open die niet óf gemeten is, óf begrensd door een
meting. De twee die alleen begrensd zijn en niet gedraaid — Nested-QAD-training
en ElasticDraft — kunnen het gat niet dichten, want de bovengrens in §3 gunt ze
hun beste geval al.

## 5. Claim boundary

Y1 is een **oracle**, geen implementatie: arm B doet het routerwerk op device en
de LRU-bookkeeping op de host, maar haalt de ids uit een capture van dezelfde run,
zodat gemeten wordt wat het wegnemen van de synchronisatie waard is. Het is geen
bewering dat een gebouwde device-side router precies dit levert. De Y1-cijfers
zijn end-to-end tokentijden op deze GPU bij capacity 72; de basislijnen liggen
hoger dan n7b's omdat de subklasse overhead toevoegt, dus alleen het **verschil**
tussen de armen telt. Y2-R1 varieert de recordgrootte van één echte NVFP4
`up_proj` met de structuur vast; het is een kostenoracle voor byte-reductie, geen
quantizer, geen kwaliteitsclaim en geen tokentijd. De tabel in §3 is aritmetiek
op gemeten componenten en is uitdrukkelijk geen resultaat.

## 6. Artefacten

`Y1Y2_READBACK_AND_BYTES_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/y1y2_readback_and_bytes.py` ·
`y1y2_readback_and_bytes.json` ·
`scripts/lightningstream_nemotron/y2r1_bytes_vs_time.py` ·
`y2r1_bytes_vs_time.json` ·
`scripts/lightningstream_nemotron/y1y2_independent_verify.py` ·
`y1y2_independent_verification.json` · `protected_verification_after_y1y2.json`
