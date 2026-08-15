# STREAMQ5-MoE P4C — fused full-overlap preregistratie

Datum: 2026-08-12. Status bij vastlegging: geen P4C-output geopend.

## Hypothese

P4A bewees bit-exacte causale full-hitcompute-overlap maar miste de testmean met
0,374 ms. P4B bewees dat een korter overlapvenster dit niet oplost. P4C houdt
daarom het volledige P4A-overlapvenster en verwijdert twee implementatiekosten:

1. één pinned metadata-H2D van 160 bytes per laag in plaats van vier transfers;
2. gate, up en SwiGLU worden per expert-rij in één kernel uitgevoerd, met exact
   dezelfde FP32-dot-reductie en `expf`-formule als P3A.

Een gemengde laag gebruikt daarmee vijf kernels: fused gate/up/SwiGLU voor
hits, down voor hits, fused gate/up/SwiGLU voor misses, down voor misses en één
reductie. De hypothese is dat dit op een derde routecapture mean <= 20,0 ms en
p95 <= 25,0 ms haalt met bit-identieke eindstates.

## Onafhankelijke evaluatie

- Routebron is P1C, niet de P3A-data van P4A of de P2B-data van P4B.
- Statische top-20 per laag wordt uitsluitend uit P1C-calibration 0:512
  bepaald. Validation 512:768 en test 768:1024 blijven gescheiden.
- Vijf domeinen; 1.280 tokens per beslissende split.
- Test wordt één keer geopend na een volledige validation-pass.
- Exacte P1D-bank, P2C 20+15/14 LRU, fysieke record-H2D, volledige cache,
  trunk- en KV-reservering en minimaal 384 MiB scratch.
- Serial-control en fused-async starten ieder token met dezelfde
  deterministische FP32-state.
- Geen toekomstige laag- of tokenroute wordt voor prefetch gelezen.

## Primaire poorten

1. serial en async hebben exact dezelfde missreeks;
2. H2D-record- en byteaantallen zijn exact;
3. eindstates zijn bit-identiek; noodgrenzen zijn daarnaast
   `max_abs <= 2e-5`, `relative_l2 <= 1e-6`;
4. alle outputs en timings zijn eindig;
5. async aggregate mean `<= 20,0 ms`, p95 `<= 25,0 ms`;
6. ieder domein mean `<= 22,0 ms`, p95 `<= 30,0 ms`;
7. serial mean `<= 35 ms` en zijn misshelling is positief;
8. async mean ligt binnen 15% van de externe 19,06-ms-voorspelling;
9. co-residency en scratchgrens worden werkelijk gehaald.

## Claimgrens

Een pass bewijst een causale fysieke expertplane onder 20 ms op een derde,
voor deze implementatie nog ongeziene routecapture. Samen met P4A toont dit dat
same-layer overlap de transferterm sterk reduceert; het bewijst niet dat die
term exact nul is. Full-modelonderdelen en autoregressieve routefeedback zijn
nog steeds niet getest.
