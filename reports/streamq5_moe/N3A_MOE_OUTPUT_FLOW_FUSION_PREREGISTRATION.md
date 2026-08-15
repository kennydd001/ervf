# N3A — MoE-output flow-fusion preregistratie

Vastgelegd vóór uitvoering.

## Hypothese

De exacte keten `Q5 down → BF16-downbuffer → routeweging → expertoptelling →
residual` kan in één kernel worden uitgevoerd. Iedere down-dot houdt de
oorspronkelijke ERVF16-reductieboom; de acht BF16-afgeronde expertwaarden worden
daarna in exact dezelfde door route-ID bepaalde volgorde gewogen en opgeteld.

## Fysieke vergelijking

- 48 lagen, acht residente fysieke Q5-experts per laag.
- Referentie: `q5_down_ervf16` plus de bestaande `weighted_residual`.
- Kandidaat: één kernel per laag, twee outputrijen per block en acht logische
  ERVF16-groepen per rij.
- Seed `120823`; per laag deterministische routegewichten, routevolgorde,
  activaties en residual.
- Correctheid over alle 48 × 2.048 state-elementen, bit-voor-bit.
- Gepaard AB/BA: validation 30 en test 120 metingen.

## Poorten

Validation opent test bij bit-exactheid en p50-ratio `<=0.98`. Test slaagt bij
p50-ratio `<=0.97` en p95-ratio `<=1.00`.

## Claimgrens

Geïsoleerde residente all-eight MoE-uitgangsplane. Geen cachemissplit,
volledige decoder, andere projection-flowketens, kwaliteit of SOTA-claim.
