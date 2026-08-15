# STREAMQ5-MoE P4B — rendezvous-async preregistratie

Datum: 2026-08-12. Status bij vastlegging: geen P4B-output geopend.

## Aanleiding en hypothese

P4A was bit-exact en versnelde de onafhankelijke P3A-test 1,402×, maar miste de
meanpoort op 20,374 ms versus 20,0 ms. De implementatie splitste gate/up,
SwiGLU én down over hits en misses. Daardoor telde een gemengde laag zeven
kernellaunches en vier kleine metadata-H2D's.

P4B splitst uitsluitend gate/up. Zodra de missrecords beschikbaar zijn,
berekent het de miss-gate/up en laat daarna alle acht expertposities samenkomen
voor één SwiGLU, één down-kernel en één reductie. Alle metadata voor een laag
wordt in één pinned 160-byte record gekopieerd. De hypothese is dat deze
`gate/up rendezvous` de causale async expertplane op een onafhankelijke
routecapture onder 20,0 ms mean brengt, met exact dezelfde outputs.

## Onafhankelijke data en vastgelegde uitvoering

- Routebron: de afzonderlijke P2B-routecapture, niet de P3A-routecapture waarop
  P4A werd ontwikkeld.
- Statische experts worden uitsluitend uit P2B-calibration tokens 0:512
  bepaald; validation 512:768 en test 768:1024 blijven gescheiden.
- Vijf domeinen, 1.280 tokens per beslissende split.
- De test wordt slechts één keer geopend na een volledige validation-pass.
- Exact P1D Q5-bank, P2C 20+15/14 LRU-policy, fysieke recordkopieën,
  4.977.623.040-byte cache, 1.541.093.376-byte trunkreservering,
  402.653.184-byte KV-reservering en minimaal 384 MiB vrije scratch.
- Gelijktijdige serial-control en rendezvous-async krijgen dezelfde
  deterministische FP32-beginstates.
- Geen route uit een toekomstige laag of token wordt gelezen voor prefetch.
- In een gemengde laag: twee gate/up-kernels, één SwiGLU, één down en één
  reductie. Bij nul misses is dit het oorspronkelijke vier-kernelpad.

## Primaire poorten

Validation en test moeten elk alles halen:

1. serial- en async-missreeksen zijn exact gelijk aan de P2C-policy;
2. fysieke H2D-record- en byteaantallen zijn exact;
3. serial- en async-eindstates zijn bit-identiek; numerieke noodgrenzen blijven
   `max_abs <= 2e-5` en `relative_l2 <= 1e-6`;
4. alle outputs en timings zijn eindig;
5. async aggregate mean `<= 20,0 ms`, p95 `<= 25,0 ms`;
6. ieder domein mean `<= 22,0 ms`, p95 `<= 30,0 ms`;
7. serial-controlmean reproduceert de bijbehorende P3A-seriële band: omdat dit
   een nieuwe routecapture is, geldt de fysieke plausibiliteitsgate
   `serial mean <= 35 ms` en moet misshelling positief blijven;
8. co-residency en minimaal 384 MiB scratch zijn werkelijk gemeten;
9. de mean ligt binnen 15% van de vóór P4A aangeleverde 19,06-ms-voorspelling.

## Claimgrens

Een pass bewijst een gerepliceerde, causale, fysiek gemeten expertplane onder
20 ms op twee gescheiden routecaptures. Het bewijst niet dat de volledige
transferterm nul is, en evenmin full-model decode. Routergewichten,
attention/trunkcompute, KV-mutatie, embedding/head/sampling, autoregressieve
routefeedback en end-to-end tok/s blijven open.
