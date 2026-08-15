# H6 Co-routed Quantization Error Cancellation — preregistratie laag 26

Vastgelegd op `2026-08-10T11:51:39.0265007Z` vóór nieuwe QERC-code en vóór
inspectie van Q3-foutcovariantie of gainresultaten.

## Hypothese

Q3-expertgewichten worden normaal onafhankelijk gekwantiseerd. Door uitsluitend
de al bestaande per-output-row dequantisatieschaal van iedere routed expert
gezamenlijk te kalibreren tegen de gewogen som van de zes co-routed experts,
kan minstens 20% van de aggregate routed Q3-fout en minstens 20% van de lokale
final-logit-KL worden verwijderd zonder integercodes, bytes of kernellayout te
veranderen.

## Vaste data en control

- DeepSeek-V2-Lite revision
  `604d5664dddd88a0433dbae533b7fe9472482de0`, laag 26;
- WikiText-2-raw-v1 revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- eerste 256 validatie- en eerste 256 testtokens uit de bestaande
  layer-26-componenttrace;
- validatieposities 0–127 zijn fit-calibratie; posities 128–255 kiezen alleen
  de regularisatie; test wordt pas daarna geëvalueerd;
- natuurlijke top-6-route en ongenormaliseerde routergewichten;
- BF16 en Q3 selected expertoutputs worden in dezelfde run en batchvorm uit
  dezelfde lokale gewichten berekend;
- de officiële original-control is
  `BF16(teacher + natural_BF16_routed - natural_BF16_routed)` en moet bitexact
  teacher, KL/CE nul en top-1 één geven.

De bestaande opgeslagen Q3-output wordt alleen als batchvormregressie
gerapporteerd en is geen bitexacte control, conform de eerder waargenomen BF16
GEMM-batchafhankelijkheid.

## Fase A — foutdecompositie

Voor iedere token-slot is

`v_s = p_s · (Q3_expert_s(x) - BF16_expert_s(x))`.

We rapporteren raw per token:

- diagonal energy `Σ_s ||v_s||²`;
- aggregate energy `||Σ_s v_s||²`;
- cross term `aggregate - diagonal`;
- cancellation fraction `(diagonal - aggregate) / diagonal`.

De primaire cross-termratio is de ratio van de sommen over tokens, niet het
gemiddelde van instabiele tokenratio's. “Near zero” is vooraf gedefinieerd als
absolute cancellation fraction `<2%` op zowel validatie als test.

## Fase B — dezelfde integercodes en layout

Alle Q3-integercodes blijven vast. Een gain per outputrow vermenigvuldigt de
bestaande down-projection-dequantisatieschaal en vervangt dus een bestaande
schaalwaarde; er komt geen metadata, index, bit of kerneloperand bij.

Vaste kandidaten/controles:

1. ongewijzigd Q3;
2. onafhankelijk per expert×outputrow gefitte gain;
3. co-routed scalar gain per expert, diagnostisch;
4. primaire co-routed per-expert×outputrow gain.

Gains zijn begrensd tot `[0,75, 1,25]`. Voor de co-routed fits zijn de vaste
ridgefactoren `{1e-4,1e-3,1e-2,1e-1,1}` relatief aan de gemiddelde Gramdiagonaal.
Iedere factor wordt op validatie 0–127 gefit en op 128–255 beoordeeld. De laagste
aggregate routed MSE wint; ties kiezen de grootste regularisatie. Daarna wordt
dezelfde factor één keer op alle 256 validatietokens herfit en onveranderd op
test toegepast. Geen testgestuurde gain, bound, methode of alpha.

De per-row co-routed least-squaresproblemen zijn per outputkanaal exact
gescheiden. Clamping gebeurt na de ridge-oplossing en wordt expliciet
gerapporteerd. Een later blockwise floor/ceil-roundingonderzoek opent alleen als
deze schaalproef positief is.

## Kwaliteit en gates

Aggregate foutreductie is
`1 - MSE(gained_routed, BF16_routed) / MSE(Q3_routed, BF16_routed)`.
Finale layer-26-metrics gebruiken volledige vocabulaire KL/CE/top-1 en
10.000× gepaarde 128-token-blockbootstrap, seed `20260810`.

De layer-26-screen is alleen positief als:

1. primaire per-row gain aggregate foutreductie `≥20%` op de validatie-
   selectieslice én op test;
2. teacher→candidate-KL-reductie versus Q3 `≥20%` op volledige validatie én
   test;
3. geen extra schaalwaarden/bytes, identieke tensorvorm en geen nieuwe
   kerneloperand;
4. exact original-control slaagt en alle gains/outputs zijn finite.

Harde falsificatie: cross terms zijn near-zero op beide splits, primaire
testfoutreductie `<10%`, test-KL verslechtert, of exact-control faalt. Een
uitkomst tussen 10% en 20% is inconclusief negatief en opent geen spread.

Alleen een positieve screen opent een nieuwe preregistratie voor lagen
1/13/23/26 en instructie/code-transfer, daarna modelbrede gaincalibratie en de
vereiste `≥20%` held-out full-depth-KL-reductie. Alternatieve CRCQ-routes worden
alleen gecombineerd als QERC én die lijn onafhankelijk positief zijn; H1 is
downstream gefalsificeerd en wordt hier dus niet gecombineerd. Geen
runtime-/snelheidsclaim zonder fysieke Q3-kernel.
