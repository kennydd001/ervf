# N3A2 — attention projection-flow-fusie preregistratie

Datum: 2026-08-12. Vastgelegd vóór correctness- of timinguitvoer.

## Hypothese

De bestaande decoder materialiseert één exacte input-RMSNorm en gebruikt die
voor Q, K en V, maar lanceert daarna drie afzonderlijke Q8-projectiekernels,
een Q/K-norm+RoPE-kernel en een V-KV-writekernel. Twee exacte fusievormen kunnen
launch- en tussenstapkosten verminderen zonder een BF16-grens te verwijderen:

1. `concat_qkv`: één ERVF-16-grid voor de aaneengesloten logische Q-, K- en
   V-rijruimtes, gevolgd door de bestaande Q/K-norm+RoPE en V-write;
2. `head_flow`: één block per Q- of KV-head dat de 128 Q8-projectierijen in
   acht ERVF-16-golven berekent en daarna binnen hetzelfde block de bestaande
   Q/K-norm, RoPE en K/V-KV-write uitvoert.

De input-RMSNorm blijft één afzonderlijke kernel en dus één gedeelde,
gematerialiseerde BF16-grens voor Q/K/V. Een echte RMSNorm→projection-fusie
over onafhankelijke CUDA-blocks vereist een gridbarrière of herberekent de norm
per outputrij en valt daarom buiten deze exacte componentproef.

## Exacte semantiek

- fysieke P6A/P13 Q8-devicebank, alle Q/K/V-records van 48 lagen;
- fysieke BF16 input-, Q- en K-normgewichten uit dezelfde bank;
- vaste FP32-toestanden met seed `120824`;
- Q8 ERVF-16-reductievolgorde blijft identiek;
- projectieoutput wordt exact naar BF16 afgerond vóór Q/K-norm of V-write;
- Q/K-norm bewaart beide bestaande BF16-afrondingen;
- RoPE bewaart BF16 cos/sin, product-, som- en eindafrondingen;
- K en V worden als dezelfde BF16-bits naar de fysieke 4096-context-KV-layout
  geschreven.

Referentie per laag: `rmsnorm`, drie `q8_ervf16`,
`qk_norm_rope_write`, `write_v`.

## Partities en correctheid

- Validationpositie: 1237; testpositie: 3079.
- Correctness vóór timing op validation: alle Q-, K-, V-outputbits en alle
  geschreven K/V-cachebits van alle 48 lagen moeten exact gelijk en eindig
  zijn. Dat zijn 294.912 FP32-uitgangen en 49.152 BF16-KV-elementen per arm.
- De geselecteerde arm wordt vóór test opnieuw exact gecontroleerd op de
  testpositie.

## Meetprotocol en poorten

- Validation: 5 warmups en 30 afwisselend geroteerde metingen van referentie,
  `concat_qkv` en `head_flow`.
- Selectie: laagste p50 onder volledig bitexacte kandidaten.
- Test opent alleen bij validation-p50-ratio `<=0.98`.
- Test: 10 warmups en 120 AB/BA-afwisselende paren.
- Pass: test-p50-ratio `<=0.97`, test-p95-ratio `<=1.00` en bitexactheid.

## Claimgrens

Een pass bewijst alleen winst op de residentiële 48-laagse attention-inputflow
op deze GPU. Attention scores/values, expertpad, H2D, volledige decoder,
kwaliteit, cross-GPU en SOTA vallen buiten de claim.
