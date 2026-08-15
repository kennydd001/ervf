# CORETAIL-MoE onderzoekslog

## 2026-08-11 — bronprecondition en locked16-formaat gelockt

De full-bank P0 kan niet rechtstreeks starten: 16/6.144 canonieke GPTQ-experts
zijn aanwezig. P0A bouwt één daadwerkelijk core- en tailbestand over die
locked16, met volledige byteboekhouding en bit-exacte decoder. Geen RTN-fallback
en geen full-bankclaim uit extrapolatie.

## 2026-08-11 — locked16 werkelijk gebouwd en onafhankelijk geverifieerd

De fysieke codec bevat 75.497.472 echte GPTQ-codes in 48 matrixrecords.
Core meet 1,743924 bpp en tail 0,252170 bpp; alle codes en BF16-scalebits
reconstrueren exact. Een tweede decoder controleerde hashes, headers, offsets,
checksums, uitlijning, bronvergelijking en rekenkunde: 28/28 geslaagd.

De lineaire full-bankprojectie valt binnen de drie geheugengates, maar blijft
diagnostisch. Door 16/6.144 broncoverage is de officiële P0 niet behaald en
blijft de fused-kernel-P1 gesloten.

## 2026-08-12 — volledige bronbank en fysieke CORETAIL-P0 geslaagd

De nieuwe pure-GPTQ-bank is onafhankelijk geverifieerd: 21/21 broncontroles
over 6.144 experts, 18.432 matrices, 28.991.029.248 codes en 226.492.416
BF16-schalen. Daarmee verviel de locked16-bronblokkade.

Het vooraf vastgelegde CORETAIL-formaat is vervolgens werkelijk over de hele
bank gebouwd. De core meet 5,883217 GiB, de tail 0,845718 GiB en de volledige
representatie 1,993759 bpp. De residentformule inclusief INT4-trunk, 4K
BF16-KV-cache en 0,75 GiB reserve meet 7,725844/7,959961 GiB.

Een onafhankelijke decoder reconstrueerde alle 28.991.029.248 codes en alle
BF16-schaalbits exact en sloot 26/26 controles. CORETAIL P0 is daarom officieel
`p0_pass`. Dit bewijst de fysieke representatie en geheugenfit op deze Qwen-bank;
P1-kernelsnelheid, modelkwaliteit, end-to-end tok/s en tweede-familiegeneralisatie
blijven afzonderlijke open claims.

## 2026-08-12 — exacte NVRTC-kernel P1 geslaagd

Na een vooraf gelockte 72-matrixbenchmark is dezelfde vaste CUDA-kernel getest
op gate/up/down voor acht werkelijk gerouteerde experts uit lagen 0, 24 en 47.
Alle 72 gevallen bleven binnen de fouttolerantie. De aggregate CORETAIL-
throughput meet 33,319 Gweight/s op p50 en 30,738 Gweight/s op de conservatieve
p95, beide boven de preregistreerde 27,2-Gweight/s-gate.

Over 5.120 bevroren routed tokens bedraagt de conservatieve p95-omvang van de
geselecteerde row-aligned tail plus offsets 74.749.072 bytes. Een pinned H2D-
kopie daarvan meet 3,314 ms p95 tegenover de 33,3-ms-gate. De runtime houdt de
tail eenmaal gedecomprimeerd in host-RAM; de conservatieve extra hostcache is
1.193.579.789 bytes.

De vijf P1-gates slaagden en een afzonderlijke audit herberekende provenance,
72 correctheidsgevallen, throughput, routed tailverdeling en gatebeslissingen:
13/13 controles. Op basis van 1.811.939.328 routed expertgewichten per token is
de geprojecteerde expert-plus-tailterm 57,624 ms p50 en 62,262 ms p95. Dit is
nog geen end-to-end tok/s: trunk, routing, activaties, scheduling en volledige
modelkwaliteit moeten in P2 en een geïntegreerde wall-clocktest volgen.
