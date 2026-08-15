# H1 CRCQ — volledige 59.136-kandidatenoracle

Datum: 2026-08-10  
Status: `full_oracle_positive`  
Machineleesbaar resultaat: `reports/craft_moe/crcq_full_oracle.json`

## Vaste uitbreiding

Na de vooraf toegestane positieve top-32-screen zijn per token alle
`924×64=59.136` combinaties van top-12-kies-6-route en Q3/Q4-masker met exacte
volledige-vocabulaire-KL geëvalueerd. Model, datawindows, quantizer,
routergewichten, BF16-deltapatch, batchgrootte, 1%-all-Q4-doel en gates zijn
vooraf ongewijzigd vastgelegd.

## Resultaat

| Split | Natural 3→4 | Top-32 joint | Volledige joint | Gem. bit | Reductie vs natural |
|---|---:|---:|---:|---:|---:|
| Validatie, 256 | 20,313% | 11,263% | **9,831%** | **3,0983** | 51,60% |
| Test, 256 | 22,461% | 14,128% | **12,240%** | **3,1224** | 45,51% |

De volledige ruimte verbetert de top-32-screen met nog 1,432 procentpunt op
validatie en 1,888 punt op test. Alle drie gepreregistreerde full-gates slagen:

- beide puntwaarden zijn ≤15%;
- de volledige ruimte is op beide splits niet slechter dan top-32;
- directe schedule-KL reproduceert de DP binnen `1e-6` (validatie-afwijking
  `3,51×10⁻⁸`, test exact `0`).

De 95%-sequence-block-bootstrap voor de volledige upgradefractie is
`9,245–10,612%` op validatie en `10,677–15,169%` op test. De testbovenkant ligt
nog 0,169 procentpunt boven de gate. Het oracleplafond is sterk, maar de kleine
exploratieve window is geen confirmatie.

## Eindgedrag bij het Q4-kwaliteitsbudget

| Metric | Validatie | Test |
|---|---:|---:|
| Teacher→candidate-KL | 0,002699 | 0,004254 |
| Relatieve CE-delta | +0,528% | −0,010% |
| Top-1-overeenkomst | 98,44% | 98,44% |
| Gekozen route in BF16: KL | 0,001514 | 0,001901 |
| Alternatieve-routefractie | 94,53% | 94,14% |

Het effect komt dus niet van enkele uitzonderingen: de oracle kiest voor bijna
alle tokens een alternatieve route, terwijl 156 validatie- en 141 testtokens
helemaal geen Q4-upgrade nodig hebben. Routevrijheid en quantisatieresidual
compenseren elkaar aantoonbaar beter dan natural-route-bitselectie.

## Controles en reproduceerbaarheid

- Natuurlijke BF16 geeft op beide splits exact KL `0`, top-1 `1` en CE-delta
  `0`.
- Natural Q3-KL reproduceert de top-32-run exact; natural Q4 verschilt op test
  slechts `3,71×10⁻¹⁰`. De natural upgradefracties zijn exact gelijk.
- 51/51 tests slagen.
- De berekening duurde 1.501,92 s vóór JSON-write; serialisatie 24,83 s.
  Piek-CUDA-allocatie was circa 1,40 GB.
- Het 1.052,5 MiB JSON bewaart alle 30.277.632 kandidaat-KL's, routes, maskers,
  DP-curves, directe tokenmetrics, bootstrapresultaten, bronhashes, gitstatus,
  commandoregel en hardware.

## Stop/go

**Go naar laag 23 met exacte lagen 24–26.** H1 heeft nu een bewezen lokaal
oracleplafond op laag 26. Dit is nog geen algemene Eureka: de keuze is
teacher-gestuurd, er is geen goedkope selector, de downstreamdynamiek is niet
getest en er is geen packed wall-clockmeting. Het volgende experiment moet
vaststellen of de route×bitwinst een eerdere interventie door de echte
modeltail overleeft.

