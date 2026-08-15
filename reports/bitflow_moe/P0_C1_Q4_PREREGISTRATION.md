# BITFLOW-MoE P0 — maximale lineaire C1/Q4-proef

**Vastgelegd:** 2026-08-11, vóór nieuwe BITFLOW-validation- of testmetrics.

De bestaande exacte DeepSeek-V2-Lite Q4-semantiek blijft ongewijzigd. Per
MoE-laag wordt sequentieel één C1-equalizer gefit:

```text
c = A RMSNorm(h_provisional_post) + B RMSNorm(m_quantized_routed)
```

`A` en `B` zijn beide 2048×2048 en worden voor evaluatie in BF16 opgeslagen.
De fit gebruikt de 1.024 eerste WikiText-train-tokens in acht onafhankelijke
blokken van 128. Lambda wordt per laag uitsluitend gekozen op de eerste 256
validationtokens uit `{1e-4, 1e-2, 1}` nadat de 4096 features door `sqrt(4096)`
zijn gedeeld. Na iedere laag worden train- en validationstudentstates met de
gekozen BF16-repair opnieuw gegenereerd.

De eerste 256 testtokens worden exact één keer geopend nadat alle 26 matrices
en lambdas vaststaan. Teacher, ongerepareerde Q4 en gerepareerde Q4 worden dan
volledig opnieuw doorgerekend.

## Gates

De lineaire tak stopt wanneer validation óf test minder dan 50% van de Q4
CE-schade herstelt. Alleen wanneer beide minstens 50% halen, mogen C0/control,
C2 en Q3 verder worden geopend.

De primaire Q4-gate blijft strenger en conjunctief:

- CE-schadeherstel minstens 70%;
- relatieve CE-toename maximaal 1%;
- teacher top-1-overeenkomst minstens 97%;
- late-layer explosieratio maximaal 2,0, gedefinieerd als de maximale repaired
  hidden-NRMSE in lagen 21–26 gedeeld door de mediaan in lagen 7–20.

Een nulnoemer in de contraction ratio wordt afzonderlijk geteld en nooit als
nulratio geïnterpreteerd. De officiële laagdecompositie en de nulrepair moeten
bit-exact dezelfde output geven. P0 doet geen runtime-, Qwen- of nieuwheidsclaim.
