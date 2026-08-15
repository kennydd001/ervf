# N3A4 — exacte O-projectie→residual-fusie preregistratie

Datum: 2026-08-12. Vastgelegd vóór N3A4 correctness of timing.

## Hypothese

De attention-outputketen schrijft nu eerst iedere Q8 O-projectie als een
BF16-afgeronde FP32-tussentensor en lanceert daarna `residual_add`, dat opnieuw
naar BF16 afrondt. Eén ERVF-16-kernel kan beide afrondingsgrenzen exact lokaal
uitvoeren en rechtstreeks de nieuwe state schrijven. Daarmee verdwijnen 48
residual-launches en de getimede tussentensor write/read.

## Fysieke scope en semantiek

- Alle 48 fysieke O-projectierecords uit de P6A/P13 Q8-devicebank.
- Iedere O-matrix heeft vorm `[2048, 4096]` en gebruikt de bestaande
  ERVF-16-reductievolgorde.
- Baseline per laag: `q8_ervf16(attention) → projected`, daarna
  `residual_add(residual, projected) → state`.
- Kandidaat: dezelfde ERVF-16-dot; lane 0 berekent achtereenvolgens
  `projected = round_bf16(dot)` en
  `state = round_bf16(residual + projected)`.
- De kandidaat schrijft de dode `projected`-scratch niet naar global memory.
- Vaste seed `120825`. De eerste 48 attention/residual-paren zijn validation;
  de volgende 48 zijn een ongeopende testset.

## Correctheid en timing

- Alle 98.304 uiteindelijke FP32-state-elementen over 48 lagen moeten
  bit-voor-bit gelijk en eindig zijn; geen tolerantie.
- Validation: 5 warmups, 30 AB/BA-afwisselende paren.
- Test opent alleen bij bitexactheid en validation-p50-ratio `<=0,98`.
- Test: nieuwe inputhelft, opnieuw bitexactheid, 10 warmups en 120
  AB/BA-afwisselende paren.
- Pass: test-p50-ratio `<=0,97` en test-p95-ratio `<=1,00`.

## Claimgrens

Een pass bewijst alleen de fysieke, residentiële O→residual-componentwinst op
deze GPU. Attention, Q/K/V, experts, H2D, volledige decoder, kwaliteit,
cross-GPU en SOTA vallen buiten de claim.
