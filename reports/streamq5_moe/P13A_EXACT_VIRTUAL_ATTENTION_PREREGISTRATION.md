# P13A preregistratie — Exact Virtual-Tile Attention + Probability Materialization

Datum: 2026-08-12. Status bij vastlegging: P12R2 geopend, geen P13A-output.

## Nieuwe hypothese

De huidige attention-scorekern lanceert één 128-thread block per head én positie
(6.291.456 blocks per token over 48 lagen bij context 4096). Bovendien berekent
de valuekern iedere softmax-exponent 128 keer opnieuw, één keer per value-
dimensie.

Twee exacte transformaties:

1. **Exact Virtual-Tile scores (EVT-8):** één warp emuleert de 128 originele
   dimensietraden en dezelfde reductieboom; acht warps verwerken acht posities
   per block.
2. **Probability Materialization (PM):** bereken en BF16-round iedere
   softmaxprobability één keer in `scores`; de valuekern hergebruikt die waarde
   in exact dezelfde p-volgorde.

## Protocol en gates

- Contexten 128, 512, 1024 en 4096; alle 48 lagen; dezelfde fysieke 402-MB KV.
- Originele P7-kernels zijn de bitreferentie.
- 3 warmups + 30 validation-iteraties; 10 warmups + 120 tests op de vooraf
  geselecteerde volledige variant.
- Score-uitvoer EVT is bitexact; attention-value-uitvoer PM is bitexact over alle
  lagen/contexten.
- Bij context 1024 p50/p95-ratio `<= 0,80`; bij 4096 `<= 0,50`.

Alleen bij pass volgt integratie in de 32-GiB 4K-decoder.

