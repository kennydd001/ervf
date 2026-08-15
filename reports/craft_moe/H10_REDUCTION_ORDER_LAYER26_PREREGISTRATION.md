# H10 Numerical Reduction-Order Compensation — preregistratie laag 26

Vastgelegd op `2026-08-10T18:40:18.0113408Z` vóór nieuwe H10-code, vóór
herberekening van Q3/Q4-expertoutputs en vóór inspectie van enige
permutatie-uitkomst.

## Hypothese en noveltygrens

Eén vaste accumulatieorde van de zes gewogen expertoutputs kan bij gelijk
Q3-weightbytebudget systematisch dichter bij de BF16-teacher komen. De vereiste
claim is niet dat verschillende floating-pointordes verschillende toestanden
kunnen produceren—dat causale verschijnsel is al aangetoond in
`arXiv:2607.28097` op DeepSeek-V4-Flash. H10 is alleen positief als één uitsluitend
op validatie gekozen orde op held-out V2-Lite-test minstens 20% van de
Q3→Q4-KL-kloof sluit zonder extra weightbytes of throughputnadeel.

## Vaste data, capture en control

- DeepSeek-V2-Lite revision
  `604d5664dddd88a0433dbae533b7fe9472482de0`, laag 26;
- WikiText-2-raw-v1 revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- eerste 256 validatie- en eerste 256 testtokens van de bestaande
  layer-26-componenttrace, elk als twee blokken van 128;
- natuurlijke top-6-route, ongenormaliseerde routergewichten en dezelfde
  routevolgorde als de opgeslagen trace;
- BF16-, symmetrische per-row Q3- en Q4-expertoutputs worden in één capture met
  dezelfde batchvorm uit dezelfde lokale gewichten herberekend;
- de officiële original-control is
  `teacher + BF16(natural_BF16_routed - natural_BF16_routed)` en moet bitexact
  de teacher opleveren.

Alle kandidaatstates worden op dezelfde manier aan de officiële teacher
verankerd. Een verschil met oudere opgeslagen Q3/Q4-output is uitsluitend een
batchvormregressiediagnostiek en geen exact-control.

## Exacte reductiesemantiek

De 720 lexicografische permutaties van slots `(0,1,2,3,4,5)` worden voor Q3 en
Q4 volledig doorlopen. Eerst ontstaan FP32 gewogen termen
`v_s = p_s * expert_s(x)`. Er zijn acht vaste schema's:

1. `fp32_sequential`: FP32-termen, FP32 nulaccumulator, zes sequentiële adds;
2. `fp32_tree`: FP32-termen, gebalanceerde boom
   `((a+b)+(c+d))+(e+f)`;
3. `bf16_operands_fp32_sequential`: iedere term eerst BF16 afgerond, exact naar
   FP32 gepromoveerd, daarna sequentieel;
4. `bf16_operands_fp32_tree`: dezelfde operands, gebalanceerde boom;
5. `bf16_sequential`: BF16-operands en BF16-rounding na iedere add;
6. `bf16_tree`: BF16-operands in de vaste boom;
7. `fp16_sequential`: FP16-operands en FP16-rounding na iedere add;
8. `fp16_tree`: FP16-operands in de vaste boom.

Na de routed reductie wordt elk schema naar BF16 gecast vóór de vaste
teacher-delta-injectie. Shared expert en alle overige modelcomponenten zijn dus
identiek. “Tree” verandert alleen de vijf addrelaties; er is geen FMA over
experttermen.

## Selectie zonder testinzage

Voor elke bitbreedte/schema/permutatie worden per token raw routed-output-MSE
tegen de same-batch natuurlijke BF16-route en de maximale order-spread
opgeslagen. De globale Q3-configuratie is de combinatie met de kleinste som van
MSE over alle 256 validatietokens. Exacte ties kiezen in deze volgorde:

`fp32_sequential`, `fp32_tree`,
`bf16_operands_fp32_sequential`, `bf16_operands_fp32_tree`,
`bf16_sequential`, `bf16_tree`, `fp16_sequential`, `fp16_tree`, en daarna de
lexicografisch eerste permutatie.

Pas na die keuze wordt dezelfde combinatie één keer op test beoordeeld. Test
kiest niets. Naast de vaste orde wordt per token de lokaal beste MSE-orde
gerapporteerd als niet-deploybare oraclebovengrens. Q4 doorloopt dezelfde 720
ordes als mechanistische control, maar mag de Q3-kandidaat niet selecteren.

Volledige-vocabulaire teacher→candidate-KL, CE en top-1 worden exact gemeten
voor natural-original, vectorized-FP32 Q3/Q4-referenties, alle acht identity-
schema's, de vaste Q3/Q4-orde en de per-token Q3/Q4-oracles. Kwaliteits-CI's
gebruiken 10.000× gepaarde 128-token-blockbootstrap, seed `20260810`.

## Primaire metric en gates

De Q3→Q4-gapclosure is

`(KL_Q3_reference - KL_fixed_Q3) / (KL_Q3_reference - KL_Q4_reference)`.

Als de denominator niet positief is, faalt de hypothese. De layer-26-screen is
alleen sterk positief wanneer:

1. vaste gapclosure `>=20%` op validatie én held-out test;
2. vaste Q3-KL niet hoger is dan Q3-reference op beide splits;
3. Q4 met dezelfde orde niet catastrofaal verslechtert en alle outputs finite
   zijn;
4. nul extra weightbytes/metadata en exact dezelfde zes termen;
5. een daarna pas geopende fysieke reducerbenchmark geen throughputnadeel
   (`<=1,05×` referentietijd) toont.

Harde falsificatie: held-out vaste gapclosure `<10%`, Q3-test-KL verslechtert,
de Q3→Q4-denominator is niet positief, de FP32-schema's zijn order-invariant of
hun beste vaste testclosure is `<10%`, of een exact-control faalt. Een resultaat
van 10–20% is inconclusief negatief en opent geen downstreamwerk.

Alleen een sterke layer-26-screen opent een nieuwe preregistratie met minimaal
1.024 validationtokens, de fysieke reducerbenchmark, laag 23 plus exacte lagen
24–26, spread en full-depth. Geen snelheid- of Eureka-claim volgt uit de Python-
oracle. De primaire studie wordt in de eindrapportage als nabije prior art
geciteerd; H10 draagt zonder nieuwe systematische held-outwinst geen noveltyclaim.
