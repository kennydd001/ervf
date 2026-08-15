# P9B statische 50%-expertpruning — preregistratie

Datum: 2026-08-12

## Hypothese

Een activatiebewuste, per-expert vaste selectie van 384 uit 768 SwiGLU-
neuronen kan iedere routed expert exact als drie kleinere dichte matrices
opslaan en uitvoeren. Daardoor halveren expertbank, expert-H2D en expert-GEMV-
gewichten zonder runtime-gather, terwijl de bestaande 2%-kwaliteitspoort
behouden blijft.

## Vastgelegde selectie

- calibratie: per domein de eerste tien 128-tokenvensters uit de immutable
  Qwen-GPTQ-supplementbundel;
- per laag en expert wordt voor neuron `j` de score
  `RMS(swiglu_activation_j) * ||down_weight[:,j]||_2` berekend;
- exact de 384 hoogste scores blijven; ties worden door neuronindex gebroken;
- dezelfde selectie compactert gate/up-rijen en down-kolommen, dus geen
  indirecte weight-gather op het hete pad;
- candidate-semantiek: deze pruning gevolgd door de reeds vaste Q5-expert- en
  INT8-trunkquantisatie.

## Evaluatie en poorten

- validation/test: de bestaande P0C 5×2 contexten per split, full depth;
- validation opent test bij relatieve CE `<=2,5%`, top-1 `>=90%`, alle 48
  lagen en exact 50% expertgewichten behouden;
- definitieve pass vereist validation én test CE `<=2,0%` en top-1 `>=90%`;
- test gebruikt exact de tijdens validation opgeslagen selectiemaskers.

Een kwaliteitspass opent pas een fysiek compact kernelbenchmark. Een
kwaliteitsfail sluit deze vaste structured-Wanda-variant zonder een
wall-clockclaim.

