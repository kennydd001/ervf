# E2GQ-MoE P0 — full-bank GPTQ-entropycensus

**Vooraf geregistreerd:** 2026-08-11  
**Status:** geen nieuwe training-routes, full-bank codes of rates geopend.

## Vraag

Generaliseert de rechtstreeks bevestigde `1,907864891374` bpp-precondition van
16 locked experts naar de volledige routed expertbank wanneer iedere expert
met exact dezelfde, vooraf vastgelegde GPTQ-regel wordt gekwantiseerd?

## Bevroren model en quantizer

- Model: `Qwen/Qwen3-30B-A3B-Base`, revisie
  `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
- Alle 48 lagen × 128 experts × gate/up/down: 6.144 experts en 18.432 matrices.
- Officiële, hash-gepinde GPTQ-prior uit GSQ commit
  `03fc16484c369e3127225615d5e03e8d3a6043e3`.
- 2 bits, symmetrisch, per-channel, MSE-range search, group size 128,
  `percdamp=0.1`, block size 128, `static_groups=False`.
- Gate/up krijgen de daadwerkelijk naar de expert gerouteerde MoE-inputs.
- Down krijgt `silu(gate(x))*up(x)` van dezelfde originele BF16-expert.
- Codes moeten exact in `{-2,-1,0,+1}` liggen; scales blijven BF16.

## Bevroren calibratie

- Bron: lokale WikiText-2 raw-v1 train-parquet met SHA-256
  `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
- Tekstregel identiek aan P1C: voeg alle niet-lege teksten met twee newlines
  samen en tokeniseer zonder special tokens.
- Neem exact de eerste 32.768 tokens en vorm 32 contexten van 1.024 tokens.
- Verwerk in contextvolgorde; geen shuffling, sampling of promptselectie.
- Minimaal 128 echte routed calibratierijen per expert. Dit getal is vast vóór
  de nieuwe routercounts worden geopend.
- Als ook maar één expert minder dan 128 rijen heeft, is P0
  `coverage_negative`. Er wordt niet aangevuld met ongerouteerde activaties,
  RTN of een andere corpusselectie binnen deze registry.

Deze expliciete regel dicht een gat in het aangeleverde agentpack: zonder een
bevroren full-bankcalibratie bestaan de 6.144 veronderstelde GPTQ-assignments
niet reproduceerbaar.

## Metingen

Voor iedere matrix, expert en laag:

- codehistogram en nulde-orde-entropie;
- histogram/entropie per rij en per group-128;
- exacte BF16-scale-bitpatronen, nulde-orde-byte-entropie en XOR/delta-entropie
  binnen rijen en tussen aangrenzende groepen;
- ideale bound, een concreet formatbudget inclusief tables/offsets/index en
  alignment, en expliciete fallbackbytes;
- calibratierijen en resourcegebruik.

P0 bouwt nog geen kwaliteitsbenchmark en opent geen validation- of testdata.

## Primaire gate

P0 is `census_positive` wanneer tegelijk:

1. alle 6.144 experts minimaal 128 routed calibratierijen hebben;
2. alle 18.432 matrices bit-exacte codes/scales opleveren;
3. een volledig gespecificeerd werkelijk bouwbaar P1-format maximaal
   `1,98 bpp` projecteert voor minstens 99% van alle routed parameters;
4. het totaal inclusief elke fallback, table, offset, expertindex, scale en
   alignment maximaal `2,0 bpp` blijft;
5. peak CUDA allocation maximaal 7,5 GiB en proces-RSS maximaal 32 GiB is;
6. twee vooraf gekozen anchors, laag 0/expert 0 en laag 47/expert 127,
   bit-exact deterministisch herhalen.

Shannon-entropie alleen kan deze gate niet halen. Alleen een positieve P0 mag
P1 openen. P0 maakt geen Eureka-, kwaliteits- of snelheidsclaim.

