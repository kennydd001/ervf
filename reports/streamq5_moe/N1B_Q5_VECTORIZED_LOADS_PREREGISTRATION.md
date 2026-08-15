# N1B — Q5 packed vector-load preregistratie

Datum: 2026-08-12. Deze hypothese en grenzen zijn vastgelegd voordat de
N1B-uitvoer werd geopend.

## Hypothese

De bestaande bitexacte ERVF-16-kernel construeert ieder 40-bit Q5-pack uit vijf
afzonderlijke byte-expressies. Minder load-instructies kunnen de fysieke
48-laagse Q5-projectieplane versnellen zonder de code-, schaal-, BF16- of
reductiesemantiek te wijzigen.

Twee vooraf gekozen varianten worden getest:

1. `aligned64x2`: één of twee natuurlijk uitgelijnde `uint64`-vensters;
2. `aligned32x2`: twee natuurlijk uitgelijnde `uint32`-vensters, gevolgd door
   een exacte byteverschuiving en een 40-bit masker.

De tweede variant leest maximaal drie extra bytes uit het aansluitende fysieke
record. Dit is veilig: elke Q5-codearray wordt direct gevolgd door zijn
schaalarray en alle records zijn minimaal tot vier bytes uitgelijnd. Er worden
geen extra bytes in de numerieke reconstructie gebruikt. Een voorafgaande
compile-only poging met `memcpy` is vervangen vóór correctness of timing omdat
de lokale NVRTC geen C-headers of `__builtin_memcpy` aanbiedt; er zijn daarbij
geen experimentele uitkomsten geopend.

## Gesloten protocol

- exact dezelfde fysieke P1D-Q5-bank en experts 0–7 van alle 48 lagen;
- vaste seed `120821`, één FP32-activatie met 4096 elementen;
- referentie: bestaande `q5_gate_up_ervf16` en `q5_down_ervf16`;
- uitvoer: gate, up en down, samen 1.376.256 FP32-elementen;
- correctness vóór timing: bitexacte FP32-bits, eindig, nul afwijkende bits;
- validation: 5 warmups per variant en 30 afwisselend geordende metingen;
- selectie: snelste correcte kandidaat op validation-p50;
- test wordt alleen geopend bij validation-p50-ratio `<= 0.98`;
- test: 10 warmups per arm, 120 AB/BA-afgewisselde paren;
- pass: test-p50-ratio `<= 0.97`, test-p95-ratio `<= 1.00` en bitexactheid.

## Claimgrens

Een pass bewijst alleen dat de geselecteerde broncode-loadvorm de geïsoleerde,
residentiële Q5-projectieplane op deze GPU versnelt. Een fail sluit deze twee
loadvormen, niet alle mogelijke Q5-layouts of toekomstige architecturen. Er is
geen end-to-end-, kwaliteits-, geheugenresidentie- of SOTA-claim.
