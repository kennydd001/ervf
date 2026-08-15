# N4B-R — exacte synthetische 80B-replicatie

Vastgelegd na de onafhankelijke N4B-audit en vóór de replicatierun.

## Reparaties en bevroren protocol

N4B-R herhaalt exact N4B met dezelfde synthetische 1.070.530.560-byte Q5-bank,
seed, widths 8/16/32, validation- en AB/BA-testomvang, maar:

1. SwiGLU is canoniek: `silu=round_bf16(g/(1+exp(-g)))`, daarna
   `out=round_bf16(silu*up)`;
2. evaluator-, N4B-, N4A-, N1C- en input-SHA256 worden opgeslagen;
3. per width wordt een SHA256 over alle 48-laagse gate/up/down-outputs
   opgeslagen;
4. alle widths moeten dezelfde digest als de canonieke width-16-output hebben;
5. timing- en 50/40/90-ms-poorten blijven identiek aan N4B.

De ongeopende test is nieuw gemeten met AB/BA; oude N4B-timings worden niet als
replicatietiming hergebruikt.

## Claimgrens

Synthetische actieve Q5-vorm plus Q8-byteprojectie. Geen checkpointpayload,
modelkwaliteit, routersporen, DeltaNet-timing of end-to-end tokens/s.
