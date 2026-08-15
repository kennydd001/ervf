# S12 — in-loop attributie van de MoE-term: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Model: `models/nemotron_3_5_lightning_v35`
Aanleiding: de S7-restpost en wat S8 daarover concludeerde.

## 1. Waarom geïsoleerd meten hier niet meer mag

S8 mat een **negatieve** niet-toegewezen term: de som der geïsoleerde
componenten was 69,287 ms tegen een gemeten token van 52,363 ms bij 262K, en
−23,2 ms bij ctx 0. De oorzaak is bekend: een geïsoleerde componentmeting
forceert een synchronisatie die de overlap in de echte lus wegneemt, dus telt
systematisch te veel. Dat was tegelijk de verklaring van de GQA-kloof die S7
openliet.

De consequentie is methodisch en bindend: elke verdere attributie moet **in de
lus** gebeuren. Anders wordt er op een spook gejaagd.

Dat is nu urgent, want de MoE-term is met 39,523 ms van de 54,3 ms de dominante
post, S9 verklaarde daarvan ~9,0 ms met GEMV-microbenchmarks, en S11 heeft net
laten zien dat de transfer het óók niet is (2,9× meer PCIe-bytes kost maar 4,8%).
Er staat dus ~30 ms open die niemand heeft gelokaliseerd, en de twee methoden die
daar tot nu toe op losgelaten zijn — geïsoleerde componentmeting en
byte-boekhouding — zijn allebei aantoonbaar ontoereikend.

## 2. De methode: replicatie in plaats van isolatie

Draai de **echte** decode-lus, onveranderd, en voeg per variant precies **één
extra aanroep** toe van één component, op dezelfde plek in dezelfde stream, met
uitvoer naar een kladbuffer. De uitvoer van het model verandert niet. Het
verschil in end-to-end tokentijd is de **marginale in-lus kosten** van die
component.

```
marginale kosten van X = p50(lus + 1× extra X) − p50(lus)
```

Dit is precies wat een isolatiemeting niet geeft: het getal is gemeten mét de
overlap die er in werkelijkheid is, want de rest van de lus draait eromheen.

Waarom replicatie en niet weglaten: weglaten verandert de uitvoer, waardoor de
identiteitspoort vervalt en het gemetene niet meer hetzelfde model is.
Replicatie laat de uitvoer bit-identiek en is daarmee toetsbaar.

**Waarom dit een ondergrens oplevert, vooraf erkend.** De tweede aanroep vindt
delen van zijn data warm in L2, en extra rekenwerk aan het einde van een laag
geeft de copy-stream meer tijd om zich te verstoppen. Beide effecten drukken het
gemeten verschil omlaag. Elk getal uit deze fase is dus een **ondergrens** op de
marginale kosten, en wordt zo gerapporteerd.

## 3. Armen

Zeven armen, één proces, één modelload, in deze volgorde:

| arm | wat er per MoE-laag extra gebeurt |
|---|---|
| `base1` | niets (de echte lus) |
| `up` | 6× extra `gemv_into` van het routed `up_proj` uit het cacheslot |
| `down` | 6× extra `down_masked_into` (panel_scan + gather + masked GEMV + reduce) |
| `router` | 1× extra `_route_device` inclusief de device→host-readback |
| `shared` | 1× extra shared-expert (twee fused GEMV's) |
| `accum` | 6× extra `accumulate_into` |
| `base2` | niets (herhaling) |

De probe wordt geïmplementeerd in een **subklasse** van `LightningRuntime` in het
runnerscript, die `super()._moe_cached()` aanroept en daarna het extra werk doet.
`runtime.py` wordt voor deze fase niet aangeraakt, zodat de gemeten lus
aantoonbaar de echte lus is en niet een variant ervan.

Contexten: **0 en 262.100**, zelfde warm-up- en sampleprotocol als
`n7b_cached_decode.py`.

Capacity **70** in alle armen, niet 72: de probe heeft ~3 MB kladruimte nodig en
bij capacity 72 is er 0,000 GiB vrij. De absolute tok/s van deze fase is daarmee
**niet** vergelijkbaar met n7b; alleen de verschillen tussen armen tellen, en die
delen allemaal dezelfde capacity.

## 4. Poorten

- **G-S12-C1 — semantiek.** De generatie is in alle zeven armen bit-identiek
  (2 prompts × 32 tokens). Een probe die de uitvoer verandert meet iets anders
  dan de lus en zijn getallen worden niet gerapporteerd.
- **G-S12-D1 — drift.** `|base2 − base1|` per context moet **kleiner** zijn dan
  de kleinste gerapporteerde marginale kosten. Marginalen die kleiner zijn dan de
  drift worden gerapporteerd als "onder de ruisvloer" en krijgen geen waarde
  toegekend.
- **G-S12-S1 — sanity.** De som van de marginalen mag de gemeten MoE-term uit S8
  (39,523 ms bij 262K) niet overschrijden. Gebeurt dat wel, dan is de replicatie
  superlineair en is de attributie ongeldig; ze wordt dan als mislukt
  gerapporteerd en niet herschaald.

Poorten worden na het zien van het resultaat niet verruimd.

## 5. Wat deze fase niet doet, expliciet

- **Geen omrekening naar aandelen.** Marginale kosten zijn geen "percentage van
  de MoE-term". Overlappende componenten hebben marginalen die niet optellen tot
  het geheel, en dat is een eigenschap van de lus, geen meetfout.
- **Geen omrekening naar tok/s.** Registry, `forbidden_hypotheses`, laatste regel.
- **Geen benoemde restpost.** Wat de marginalen niet dekken, wordt als getal
  gerapporteerd en niet "overhead" of "glue" genoemd — dezelfde regel die S8
  hanteerde toen de restpost negatief bleek.
- **Geen optimalisatie.** Deze fase bouwt niets en stelt niets voor. Zij lokaliseert.

## 6. Artefacten

`scripts/lightningstream_nemotron/s12_in_loop_attribution.py` ·
`s12_in_loop_attribution.json` ·
`scripts/lightningstream_nemotron/s12_independent_verify.py` ·
`s12_independent_verification.json` · `protected_verification_after_s12.json` ·
rapport met claim boundary.

## 7. Claim boundary van dit document

Geen meting, geen resultaat.
