# H7 Route-Local Sparse Coreset — exploratieve preregistratie

Vastgelegd op 2026-08-10 vóór uitvoering van de H7-code. Deze registratie is
exploratief/oraclegericht; zij opent geen nieuw confirmatory testvenster.

## Vaste input

- DeepSeek-V2-Lite Base, commit
  `604d5664dddd88a0433dbae533b7fe9472482de0`;
- WikiText-2-raw-v1, commit
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- laag 26;
- eerste 256 validatie- en eerste 256 testtokens uit de bestaande
  `layer26_dynamic_precision_components.safetensors`-trace;
- test wordt alleen als vaste replicatiesplit uitgevoerd, niet gebruikt voor
  solver- of gatekeuze.

## Exact-control en counterfactualpatch

De zes BF16-expertoutputs worden opnieuw uit de gepinde laag berekend en met de
originele, niet-hergenormaliseerde routergewichten opgeteld. Om numerieke
verschillen met de officiële MoE-reductieorde niet als coresetfout te tellen,
wordt iedere kandidaat als een routed-outputdelta op de officiële teacherstate
geïnjecteerd:

```text
candidate_hidden = BF16(official_teacher + fitted_routed − manual_top6_routed)
```

Voor de originele top-6 is de delta exact nul. Deze control moet daarom
teacher→candidate-KL `0`, top-1 `1` en CE-delta `0` geven.

## Vooraf vastgelegde families

Voor iedere cardinaliteit `k=1..5` worden alle `C(6,k)` subsets geprobeerd:

1. originele routercoëfficiënten, alleen experts droppen;
2. vrije least squares;
3. exacte cardinaliteitsbegrensde NNLS via enumeratie van alle positieve
   least-squares active sets;
4. box-constrained least squares met `0 ≤ αᵢ ≤ 2pᵢ`, opgelost met vaste
   coordinate descent;
5. afzonderlijke baselines: originele gerankte top-k, originele top-1 en de
   originele top-1 met optimaal niet-negatief scalair gewicht.

Subset- en coëfficiëntkeuze minimaliseren uitsluitend routed-output-L2. De
volledige-vocabulaire-KL wordt pas daarna gemeten en stuurt de keuze niet.

## Primaire gate

De primaire methode is NNLS en de primaire split is validatie. H7 is
`oracle_positive` wanneer bij lokale teacher→candidate-KL ≤ `0,001`:

- de mediaan van de minimale benodigde `k` hoogstens 3 is; **of**
- de hogere empirische p95 van minimale `k` hoogstens 4 is.

Dezelfde vaste uitkomst wordt daarna op de bestaande testreplicatie
gerapporteerd. Vrije least squares is alleen een optimistische bovengrens;
zij kan de primaire gate niet alleen laten slagen.

## Falsificatie en vervolg

H7 wordt gefalsificeerd wanneer bij NNLS zelfs `k=5` voor meer dan 25% van de
validatietokens boven KL `0,003` blijft. Alleen bij een positieve primaire gate
volgt een laag-23-interventie met exacte lagen 24–26. Een deployabilityclaim
vereist later een goedkope coëfficiëntregel die minstens 70% van de
oracle-bytewinst behoudt.

## Uitvoeringsfasen

1. unit tests en exact-control;
2. smoke op 32 validatietokens;
3. vaste oracle op 256 validatie- en 256 testtokens;
4. ongewijzigde gateberekening en append-only rapportage.
