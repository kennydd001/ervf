# H1 Counterfactual Route-Coded Quantization — exploratieve preregistratie

Vastgelegd op 2026-08-10 vóór uitvoering van H1-code of H1-metriekinspectie.
Dit is een exploratieve oracle-screen en opent geen nieuw confirmatory venster.

## Vaste input en controle

- DeepSeek-V2-Lite Base, commit
  `604d5664dddd88a0433dbae533b7fe9472482de0`;
- WikiText-2-raw-v1, commit
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- laag 26, eerste 256 validatie- en eerste 256 bestaande testtokens;
- top-12 experts met alle `C(12,6)=924` routes;
- originele, niet-hergenormaliseerde routergewichten voor de gekozen experts;
- symmetric per-output-row Q3 en Q4, identiek aan de bestaande
  dynamic-precision-oracle.

BF16-, Q3- en Q4-expertoutputs worden opnieuw uit de gepinde laag berekend.
Iedere kandidaat wordt als delta op de officiële teacherstate geïnjecteerd:

```text
candidate_hidden = BF16(official_teacher + candidate_routed - natural_BF16_routed)
```

De natuurlijke BF16-route moet daardoor exact KL `0`, top-1 `1` en CE-delta
`0` geven. De bestaande testwindow is alleen een vaste replicatie; zij wordt
niet gebruikt om shortlistgrootte, marges of gates te kiezen.

## Vaste staging

1. Unit-tests en een 32-token-validatiesmoke.
2. Stage A: exacte volledige-vocabulaire teacher→candidate-KL voor alle 924
   all-Q3-routes per token.
3. Stage B: behoud exact 32 routes per token: de 32 laagste Stage-A-KL's, met
   verplichte opname van de natuurlijke route door vervanging van nummer 32
   wanneer nodig.
4. Stage C: exacte volledige-vocabulaire-KL voor alle 64 Q3/Q4-maskers op die
   32 routes, dus 2.048 kandidaten per token.
5. Los voor zowel de natuurlijke route als de gezamenlijke top-32-ruimte de
   globale upgrade-rate–distortioncurve exact op met dezelfde discrete dynamic
   program als de bestaande 3→4-bit-oracle.

De kwaliteitsdoelwaarde is vooraf vastgezet op `1,01 ×` de gemiddelde KL van
de natuurlijke all-Q4-route op dezelfde split. Dit is dezelfde 1%-definitie
waarmee de bestaande natuurlijke testoracle `23,812%` upgrades rapporteerde.
Route- en maskerselectie gebruiken alleen teacher-KL en zijn dus een oracle,
geen deploybare selector.

De volledige `924×64=59.136` route-maskerruimte wordt **niet** geopend op een
negatieve shortlistscreen. Zij krijgt een afzonderlijke append-only run en
resourcecheck alleen wanneer de top-32-screen op validatie sterk positief is.

## Baselines en ruwe uitvoer

- exacte natuurlijke BF16-control;
- natuurlijke all-Q3 en all-Q4;
- beste alternatieve all-Q3-route, met de natuurlijke route uitgesloten;
- natuurlijke exact-dynamische 3→4-bitcurve;
- gezamenlijke top-32 route+bitcurve;
- BF16-KL van gekozen routes als diagnostiek voor echte route-equivalentie.

Alle Stage-A-KL's, Stage-C-KL's, shortlistindices, gekozen routes/maskers,
upgradeplannen en eindmetrics blijven in het machineleesbare JSON-resultaat.

## Vooraf vastgelegde gates

Een criterium is `strong_positive` wanneer het op validatie én in dezelfde
richting op de vaste testreplicatie slaagt:

1. gezamenlijke minimum-upgradefractie bij all-Q4-kwaliteit is hoogstens 15%;
2. equivalente gemiddelde actieve precisie is hoogstens 3,15 bit;
3. de beste alternatieve all-Q3-route sluit minstens 50% van de gemiddelde
   natuurlijke Q3→Q4-KL-gap.

De route-as is negatief bij minder dan 10% gap closure op beide splits. De
top-32 joint-as is negatief wanneer meer dan 25% upgrades nodig blijft op beide
splits. Als beide assen negatief zijn, luidt het oordeel `screen_negative`;
anders `inconclusive`. Een negatieve top-32-screen wordt niet aangeduid als
een wiskundige falsificatie van de ongeopende volledige 59.136-ruimte.

Alleen een sterke validatie-uitkomst mag de volledige route-maskerzoekruimte en
daarna een laag-23-interventie met exacte lagen 24–26 openen. Een
deployabilityclaim vereist later een teacher-free route-/bitselector en een
fysieke packed runtime.

