# ERGV-C2 preregistratie — generated physical-bank autotuner

Datum: 2026-08-12. Status bij vastlegging: de C2-CUDA-module is nog niet
gecompileerd, de fysieke banken zijn niet geladen en er is geen C2-timing- of
correctheidsoutput geopend.

## Hypothese

Een automatisch uit `ExactReductionIR` gegenereerde width-search over
`4/8/16/32/64` kan de handgeschreven P7-width-16-kernels mechanisch
reproduceren en op minstens één volledige fysiek residentiële projectiefamilie
ten minste 2% p50 winnen zonder p95-regressie. De andere bank mag maximaal 2%
regresseren.

## Bevroren inputs

- Q8: de bestaande volledige P6A-devicebank met 241 records, gegroepeerd per
  projectienaam `head/k/o/q/router/v`.
- Q5: experts 0–7 van alle 48 lagen uit de bestaande fysieke bank; gate/up en
  down worden afzonderlijk geselecteerd en als één Q5-plane getest.
- Activatie: één vaste FP32-vector uit NumPy `default_rng(120844)`.
- Geen H2D-kopieën of allocaties in getimede gebieden.
- Eén non-blocking CUDA-stream en CUDA-events.

De bank- en bron-SHA's worden in ieder geldig resultaat vastgelegd.

## Kandidaten en referenties

Generated kandidaten:

- widths `4,8,16,32,64` voor Q8, Q5 gate/up en Q5 down;
- 256 fysieke threads per block;
- `rows_per_block = 256 / width`;
- width 64 gebruikt de door de IR vereiste gedeelde stride-32-stap, gevolgd
  door warp-shuffles `16,8,4,2,1`.

Referenties:

1. P6B 256-thread shared-memorykernel voor bitcorrectheid;
2. handgeschreven P7, uniform width 16;
3. handgeschreven N1C met de reeds bevroren keuzes:
   `head16,k64,o16,q16,router64,v64,gate_up8,down8`.

## Compilefase

De eerste run is uitsluitend `--phase compile`. Ze genereert alle helpers en
wrappers, compileert één CUDA-module en schrijft alleen compileduur,
source-digests en kernelresources. Ze laadt geen fysieke modelbank en voert
geen kernels of timing uit. Een compilefout mag vóór de meetfase uitsluitend
mechanisch worden gerepareerd; iedere mislukte compile wordt apart bewaard.

Na een compile-pass blijft de timingfase gesloten tot expliciete GPU-toestemming.

## Correctheidspoort

Vóór timing moet de ongebruikte fysieke output van iedere generated width voor
de volledige Q8- en Q5-plane worden vergeleken met P6B. Tevens worden generated
width 16 met manual P7 en de generated bevroren N1C-graph met manual N1C
vergeleken.

Iedere vergelijking vereist:

- nul verschillende FP32/BF16-outputbits;
- maximale absolute fout nul;
- alle outputs eindig.

Een fout sluit C2 onmiddellijk; er volgt dan geen performanceverdict.

## Validation en selectie

- Per Q8-projectienaam en Q5-deelplane: 3 warmups en 15 metingen per width.
- De volgorde roteert per ronde en wordt in oneven ronden omgekeerd.
- Alleen bitexacte widths zijn selecteerbaar.
- Laagste validation-p50 wint. Binnen 0,5% van de beste waarde wint width 16
  indien aanwezig; anders de kleinste width.
- De resulterende generated Q8- en Q5-graphs worden vóór test bevroren.

## Ongeopende test en AB/BA

Na 10 warmups per variant worden per bank 120 gepaarde CUDA-eventmetingen
uitgevoerd. Even ronden zijn AB, oneven ronden BA. Er zijn twee paren:

1. manual P7-width-16 versus de bevroren generated graph;
2. manual N1C versus dezelfde bevroren generated graph.

Generated width 16 versus manual P7 wordt daarnaast met dezelfde
correctheidsdata als de mechanische P7-reproductie gerapporteerd; de primaire
performancepoort gebruikt de geselecteerde generated graph.

## Primaire passpoort

C2 slaagt alleen als:

1. alle correctheidspoorten slagen;
2. manual P7 door generated width 16 bitexact wordt gereproduceerd;
3. tegenover manual P7 heeft minstens één van `Q8` of `Q5`:
   `generated_p50 / p7_p50 <= 0,98` en
   `generated_p95 / p7_p95 <= 1,00`;
4. op de andere bank zijn zowel p50- als p95-ratio `<= 1,02`.

De vergelijking met manual N1C is een verplichte gerapporteerde paritycontrole,
maar geen afzonderlijke C2-passpoort: N1C doorzocht dezelfde widths handmatig.
Een winst tegenover P7 mag daarom alleen als compilerreproductie/autotuning
worden beschreven, niet als een nieuwe kernel boven N1C.

## Bewijsgrens

C2 betreft één modelbank, één activatie, één GPU en lokale projectieplanes. Het
is geen end-to-end tok/s-, tweede-model-, tweede-architectuur-,
publieke-baseline- of nieuwheidsclaim. Alleen een vooraf geregistreerde latere
fase mag gegenereerde nieuwe schedule-opties toevoegen.
