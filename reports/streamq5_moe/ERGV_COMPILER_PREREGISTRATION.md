# ERGV-C0 preregistratie — restricted exact-reduction compiler

Datum: 2026-08-12. Status bij vastlegging: er is nog geen ERGV-C0-uitvoer
geopend of aangemaakt.

## Doel

Deze proef bouwt een klein, lokaal compilerprototype rond het bewezen
P6B/P7/N1C-reductiepatroon. De compiler mag de fysieke CUDA-threadindeling
wijzigen, maar niet de geordende FP32-reductie-DAG, bronvolgorde, casts of
FMA-policy.

De scope is bewust beperkt tot de bestaande Q8- en Q5-GEMV-semantiek en
subwarpbreedtes `4, 8, 16, 32, 64`. Dit is een formele en CPU-geteste
correctheidspoort; er wordt in deze fase geen GPU-code gecompileerd of getimed.

## Vastgelegde bronimplementaties

- P6B-referentie:
  `scripts/streamq5_moe/run_p6a_end_to_end_decode.py`;
- handgeschreven P7:
  `scripts/streamq5_moe/run_p7b_ervf_kernel.py`;
- N1C-uitbreiding en keuzes:
  `scripts/streamq5_moe/run_n1c_generalized_exact_reduction_autotuner.py`.

Hun SHA-256-digests worden in het resultaat opgenomen.

## Restricted ExactReductionIR

De IR representeert minimaal:

1. 256 benoemde logische accumulatoren;
2. de geordende add-edges van de P6B-boom, met strides
   `128, 64, 32, 16, 8, 4, 2, 1`;
3. de uiteindelijke BF16-round/cast;
4. de FMA-policy van de bronkernel;
5. de bronlaadvolgorde per virtuele accumulator;
6. de mapping van virtuele accumulator naar fysieke lane;
7. per add-node de uitvoeringsfase: lane-local, warp-shuffle of de expliciete
   cross-warpstap voor breedte 64.

Q8 gebruikt kolommen als logische work-items. Q5 gebruikt 40-bit packs als
work-items en bewaart daarbinnen de seriële volgorde van acht 5-bit codes.

## Mechanische correctheidspoort

Een schedule is geldig als:

- de gematerialiseerde, geordende operatorboom isomorf is met de referentie;
- alle leaf-ID's precies eenmaal voorkomen;
- casts, FMA-policy en bronvolgorden gelijk zijn;
- iedere virtuele accumulator precies op lane `id mod width` staat;
- breedte 64 precies één cross-warp stride-32-fase bevat;
- breedtes tot en met 32 alleen lane-local folds en begrensde shuffles
  gebruiken.

De verifier moet daarnaast doelbewust gewijzigde bomen, bronvolgorden, casts,
FMA-policy's en lane mappings afwijzen.

## CPU-testset en passpoort

De testset omvat:

- IR-invarianten voor Q8 en Q5;
- isomorfie voor alle vijf breedtes en beide quantisatiefamilies;
- bitvergelijking van de referentie en ieder schedule op vaste random en
  adversariële FP32-leafwaarden;
- mechanische audit dat P7 de breedtes `8/16/32` en N1C de breedtes
  `4/8/16/32/64` volgens het verwachte schema bevat;
- deterministische CUDA-codegeneratie voor Q8 en Q5;
- representatie van de bevroren N1C-keuzes
  (`head16,k64,o16,q16,router64,v64,gate_up8,down8`);
- negatieve mutatietests.

ERGV-C0 slaagt alleen als alle tests slagen. Een enkele fout sluit de CPU-gate.

## Bewijsgrens

Een pass bewijst dat het restricted prototype de huidige handgeschreven
reductiegrafen mechanisch kan beschrijven, verifiëren en genereren. Het bewijst
geen GPU-compileerbaarheid, bitgelijkheid op echte modelgewichten, snelheid,
generalisatie buiten deze kernels of nieuwheid. Een GPU-gate vereist een aparte
toestemming en vooraf vastgelegde meetprocedure.
