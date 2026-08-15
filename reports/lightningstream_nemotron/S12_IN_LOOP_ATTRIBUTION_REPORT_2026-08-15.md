# S12 — in-lus attributie van de MoE-term

Datum: 2026-08-15
Verdict: **De MoE-term is niet één ding. Vier componenten hebben in de lus een marginale ondergrens die samen 15,5 ms van de 39,5 ms dekt; de router-probe is niet vergelijkbaar met de rest en wordt apart behandeld. De eerste run faalde op zijn eigen driftpoort en is als zodanig gerapporteerd; de herhaling met gebracketeerde basislijnen haalt hem.**
Terminal state: `s12_in_loop_attribution_partial_router_probe_confounded`
Preregistraties: `S12_IN_LOOP_ATTRIBUTION_PREREGISTRATION_2026-08-15.md` en
`S12R1_BRACKETED_ATTRIBUTION_PREREGISTRATION_2026-08-15.md` (beide bevroren vóór hun run)

## 1. De methode, en waarom deze en geen andere

S8 liet zien dat geïsoleerd componentmeten hier systematisch te veel telt: de som
der delen was 69,287 ms tegen een gemeten token van 52,363 ms bij 262K, −23,2 ms
bij ctx 0. Een sync om een component te timen vernietigt de overlap die de echte
lus wél heeft.

Deze fase meet daarom nooit een component apart. Zij draait de **echte** lus en
voegt per variant precies **één extra aanroep** van één component toe, met
uitvoer naar een kladbuffer:

```
marginale kosten van X = p50(lus + 1× extra X) − p50(lus)
```

De probe zit in een **subklasse** van `LightningRuntime` in het runnerscript;
`runtime.py` is niet aangeraakt. De verifier controleert dat ook expliciet — het
woord `probe` komt er niet in voor — zodat de lus waartegen gemeten is
aantoonbaar de lus is die draait.

De generatie is in alle armen bit-identiek. Een probe die de uitvoer verandert
zou iets anders meten dan de lus.

## 2. De eerste run faalde op zijn eigen driftpoort

| | ctx 0 | ctx 262100 |
|---|---:|---:|
| `base1` p50 | 36,295 ms | 53,830 ms |
| `base2` p50 (na vijf probe-armen) | 38,627 ms | 58,886 ms |
| **drift** | **2,331 ms** | **5,057 ms** |

G-S12-D1 eiste dat de drift kleiner is dan de kleinste gerapporteerde marginale
waarde. Bij 262100 was dat niet zo (5,057 > 3,898), dus **de poort faalde** en
`up`, `shared` en `accum` vielen daar onder de ruisvloer. Die uitkomst staat
onveranderd in `s12_in_loop_attribution.json`.

De drift is eenzijdig — `base2` is in beide contexten trager — en de oorzaak is
zichtbaar geworden in de herhaling, die temperatuur meelogt: de GPU draait op
**86–87 °C**. De probe-armen doen meer werk en verwarmen hem; `base2` draaide na
alle vijf. Ter contrast: S11's twee identieke armen dreven 0,042 ms uit elkaar
bij dezelfde diepte. De meetlus is stabiel zodra alle armen even zwaar zijn.

**De poort is niet verruimd.** Wat veranderd is, is het schema.

## 3. De herhaling: elke probe tussen twee basislijnen

`base0 · up · base1 · down · base2 · router · base3 · shared · base4 · accum · base5`,
met de marginale waarde gemeten tegen het gemiddelde van de twee omsluitende
basislijnen, en met een **eigen lokale ruisvloer** per probe.

### ctx 262.100

| component | replicaties per MoE-laag | marginaal (ms/token) | lokale drift | |
|---|---:|---:|---:|:--:|
| `router` | 1 | +8,156 | 0,992 | ⚠ zie §4 |
| `down` | 6 | **+7,478** | 2,586 | ✅ |
| `up` | 6 | **+4,756** | 0,726 | ✅ |
| `shared` | 1 | **+3,298** | 1,257 | ✅ |
| `accum` | 6 | +0,533 | 1,146 | onder de ruisvloer |

### ctx 0

| component | marginaal | lokale drift | |
|---|---:|---:|:--:|
| `down` | **+6,144** | 4,556 | ✅ |
| `up` | **+5,245** | 0,398 | ✅ |
| `router` | +4,275 | 3,271 | ⚠ zie §4 |
| `shared` | **+3,481** | 0,267 | ✅ |
| `accum` | +1,403 | 2,388 | onder de ruisvloer |

Poorten: G-S12R-C1 (identiteit) gehaald in alle elf armen · G-S12R-S1 (som ≤ de
S8-MoE-term) gehaald · G-S12R-T1 (globale drift 4,722 ms < grootste marginaal
8,156 ms bij 262K, 4,339 < 6,144 bij ctx 0) conclusief. Verifier 99/99,
`VERIFIED`.

## 4. De router-probe is niet vergelijkbaar, en dat is een ontwerpfout van mij

De vier andere probes repliceren puur reken- en transferwerk. De router-probe
repliceert óók de **device→host-readback**, en die is een synchronisatie.

Een sync meet niet alleen zijn eigen kosten maar vernietigt ook de overlap op de
plek waar hij staat. En mijn probe staat op de verkeerde plek: aan het **einde**
van de MoE-laag, waar zes experts aan werk in de wachtrij staan, terwijl de
échte readback vroeg in de laag zit — juist daar neergezet zodat de shared expert
eroverheen kan lopen (zie de commentaarregels in `_moe_cached`). De probe draineert
dus meer wachtrij dan de echte readback doet.

Daarom: **+8,156 ms is geen "de router kost 8 ms".** Het is wat het kost om er
een tweede sync bij te zetten op een ongunstig punt. De waarde wordt hier
gerapporteerd en niet in de attributie meegeteld.

Wat het getal wél corroboreert: 8,156 ms over 23 MoE-lagen is 0,355 ms per laag,
en de runtime's eigen commentaar citeert een eerdere meting van **0,339 ms per
laag** voor precies deze readback. Twee onafhankelijke metingen, dezelfde orde.
Dat de sync duur is, staat daarmee stevig; hoeveel ervan in de huidige plaatsing
overblijft, is met deze opzet niet gemeten.

De correcte vervolgmeting is een probe met **gematchte plaatsing**: de extra
`_route_device` + readback direct achter de echte, vóór de shared expert. Aparte
preregistratie, want het is een ander ontwerp.

## 5. Wat er nu over de MoE-term bekend is

| bron | grootheid | waarde |
|---|---|---:|
| S8, gemeten | MoE-term bij 262K | 39,523 ms |
| S9, microbench × calls | up+down GEMV's, geschatte grens | ~9,0 ms |
| S11, gemeten | 2,9× meer PCIe-bytes kost | 4,8% |
| **S12-R1, in de lus** | `down` + `up` + `shared`, ondergrenzen | **15,53 ms** |
| S12-R1 | `accum` | onder de ruisvloer |

De drie schone marginalen dekken samen 15,53 ms van de 39,52 ms. Dat is méér dan
S9's microbenchmark-grens van 9,0 ms suggereerde, wat klopt: de in-lus marginaal
van `down` bevat `panel_scan`, de gather en de reductie, niet alleen de GEMV.

Wat niet gedekt is, blijft **een getal en krijgt geen naam**. Het is bovendien
een overschatting van het onverklaarde deel, want elke marginale waarde is een
**ondergrens**: de tweede aanroep vindt data warm in L2, en extra werk aan het
einde van een laag geeft de copy-stream meer ruimte om zich te verstoppen.

Wat de meting wél uitsluit: er is geen enkele component wiens replicatie de
MoE-term ook maar half verklaart. De 39,5 ms zit niet in één plek waar een
kernel-herschrijving hem weghaalt. Dat is een negatief resultaat, en het is
precies het soort dat voorkomt dat de volgende sessie een kernelproject begint op
een vermoeden — dezelfde dienst die S9's tien-regelige launch-probe bewees.

## 6. Wat deze fase niet doet

Geen omrekening naar aandelen — overlappende componenten hebben marginalen die
niet optellen tot het geheel, en dat is een eigenschap van de lus, geen meetfout.
Geen omrekening naar tok/s. Geen benoemde restpost. Geen optimalisatie: deze fase
bouwt niets en stelt niets voor.

## 7. Meetnotitie over de verifier

De S12-verifier faalde eerst op een eigen hygiënecheck: een substringzoektocht
naar "share" in de hele JSON raakte (a) de claim boundary's eigen zin dat de
marginalen géén aandelen zijn, en daarna (b) de armnaam `shared`. Beide keren was
de check verkeerd geformuleerd, niet het resultaat. Hij inspecteert nu veldnamen
met woordgrenzen. Zelfde klasse fout als S9's te strenge gelijkheidscheck bij de
blokgrootte-probe, en om dezelfde reden hier vermeld in plaats van stil hersteld.

## 8. Claim boundary

Marginale in-lus kosten, end-to-end gemeten op deze GPU bij **capacity 70** — niet
72, want de probe heeft kladruimte nodig en bij 72 is er 0,000 GiB vrij. De
absolute tokentijden van deze fase zijn daarom **niet** vergelijkbaar met n7b;
alleen de verschillen tussen armen tellen, en die delen allemaal dezelfde
capacity. Elke marginale waarde is een **ondergrens**. De router-marginaal is om
de reden in §4 geen componentkost. Niets hiervan is naar tokens per seconde
omgerekend. Geen kwaliteitsclaim, geen uitspraak over andere hardware.

## 9. Artefacten

`S12_IN_LOOP_ATTRIBUTION_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/s12_in_loop_attribution.py` ·
`s12_in_loop_attribution.json` ·
`scripts/lightningstream_nemotron/s12_independent_verify.py` ·
`s12_independent_verification.json` ·
`S12R1_BRACKETED_ATTRIBUTION_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/s12r1_bracketed_attribution.py` ·
`s12r1_bracketed_attribution.json` ·
`scripts/lightningstream_nemotron/s12r1_independent_verify.py` ·
`s12r1_independent_verification.json` · `protected_verification_after_s12.json`
