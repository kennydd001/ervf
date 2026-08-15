# Y1/Y2 — readback-eliminatie en de byte-hypothese: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Aanleiding: de twee richtingen die na X1 overblijven, plus de gedeelde premisse
onder ExactFlow A/B/C/D.

## 1. Y1 — wat is de host-round-trip waard?

Kimi's S14 mat met CUDA-events op de stream dat de MoE-lagen 27,7 ms per token
kosten en dat daarvan **`host_gap` 4,7 ms** GPU-idle is terwijl de host de
readback afhandelt, de LRU bijwerkt en kopieën uitgeeft, plus **`route` 3,5 ms**
voor een GEMV van 344 kFLOP. Samen ~8 ms per token dat geen rekenwerk is.

De oorzaak staat in `runtime.py` zelf beschreven: de expert-ids moeten naar de
host om de gepinde bank te indexeren, en elke device→host-overdracht kost een
volledige synchronisatie — 23 per token.

**Geen bouw, een oracle.** Arm B draait de échte lus, laat `_route_device` op
device volledig draaien (GEMV, sigmoid, bias, argsort — dat werk blijft in elke
implementatie bestaan), maar **leest hem niet terug**: in plaats daarvan komen de
route-ids uit een vooraf opgenomen capture van exact dezelfde run. De routes zijn
daarmee per constructie identiek en de uitvoer moet bit-identiek zijn.

Wat dit meet is de waarde van het wegnemen van de **synchronisatie**. De
LRU-bookkeeping en het uitgeven van de kopieën blijven host-werk in arm B, maar
zonder sync overlappen ze met GPU-werk in plaats van te stallen — wat een
device-side routing-implementatie ook zou opleveren.

- **G-Y1-C1:** generatie bit-identiek tussen arm A en arm B (2 × 64 tokens).
  Faalt dit, dan wordt er geen tijd gerapporteerd.
- **G-Y1-P1:** de winst bij ctx 262100 en ctx 0 wordt gerapporteerd tegen
  gebracketeerde basislijnen. Een winst telt alleen als zij de lokale drift
  overschrijdt.
- **G-Y1-S1:** de gemeten winst mag `host_gap + route` uit S14 (8,2 ms bij 262K)
  niet overschrijden. Doet zij dat wel, dan meet de opzet iets anders dan de sync
  en is zij ongeldig.

## 2. Y2 — koopt het besparen van bytes tijd?

ExactFlow A (OrbitANS), B (PathQ), C (CertiPlane) en D (Nested-QAD) delen één
premisse: minder bytes per gewicht → minder tijd. Drie metingen hebben die
premisse al aangetast — S11 (2,9× meer PCIe kost 4,8%), S12 (per-expert
marginalen 12,23 van 39,5 ms), X1 (elk record 1,59× minder vaak lezen levert
niets) — maar geen daarvan varieerde de **recordgrootte zelf**.

Y2 doet dat direct: dezelfde `gemv_nvfp4_rows` op een echt `up_proj`-record, met
de kolombreedte teruggebracht tot 100 / 75 / 50 / 25% van de bytes, structuur
verder ongewijzigd. Dat is geen semantiek — het is een **kostenoracle** dat de
vraag beantwoordt die alle vier de hypotheses moeten beantwoorden voordat er een
quantizer gebouwd wordt: *als een codec de expert-bytes halveert, hoeveel tijd
levert dat op?*

- **G-Y2-1:** halvering van de bytes moet **≥ 40%** tijd besparen om de
  byte-tak open te houden. Dat is de drempel waaronder ExactFlow's eigen
  OrbitANS-poort (12% packreductie voor doorgang, 20% sterk) niet meer dan een
  paar procent doorvoer kan opleveren.
- **G-Y2-2:** de effectieve bandbreedte wordt gerapporteerd tegen de
  device-roofline, zodat zichtbaar is of de kernel bandbreedte- of
  bezettingsgebonden is.
- Onder 20% tijdwinst bij halvering geldt de byte-tak als **begrensd**: A, B, C
  en D mogen dan nog steeds gebouwd worden, maar niet meer met doorvoer als
  motivatie zonder eerst uit te leggen waarom zij anders zouden uitpakken.

## 3. Wat hier niet gebeurt

Geen quantizer, geen training, geen wijziging aan de targetsemantiek. `runtime.py`
blijft ongewijzigd; Y1's probe zit in een subklasse in het runnerscript. Geen
enkele componentmeting wordt naar tok/s omgerekend; Y1 meet wel end-to-end
tokentijd en dáár is tok/s de eenheid van de meting zelf.

## 4. Artefacten

`scripts/lightningstream_nemotron/y1y2_readback_and_bytes.py` ·
`y1y2_readback_and_bytes.json` ·
`scripts/lightningstream_nemotron/y1y2_independent_verify.py` ·
`y1y2_independent_verification.json` · rapport met claim boundary.

## 5. Claim boundary van dit document

Geen meting, geen resultaat.
