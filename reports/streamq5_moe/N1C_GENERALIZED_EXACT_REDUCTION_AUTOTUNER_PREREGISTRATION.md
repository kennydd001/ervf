# N1C preregistratie — Generalized Exact Reduction Graph Autotuner

Datum: 2026-08-12. Registry-item: N003. Status bij vastlegging: er is geen
N1C-resultaatbestand geopend of aangemaakt.

## Aanleiding en hypothese

P7B testte alleen subwarpbreedtes `{8,16,32}`. P8A koos daaruit per
projectietype, maar de resulterende gemengde grafiek verbeterde de volledige
Q8-plane niet. N1C toetst de nog open exacte geometrieën `4` en `64` samen met
`8`, `16` en `32`. Breedte 64 gebruikt vier rijgroepen per 256-thread block en
voltooit de originele 256-thread optelboom exact via een gedeelde cross-warp
stap op stride 32, gevolgd door dezelfde warpboom.

De hypothese is dat een uitgebreidere, per-projectietype gekozen exacte
reductiegrafiek de fysiek residentiële Q8- en/of Q5-projectieplane ten minste
3% versnelt tegenover de huidige uniforme ERVF-16-kernel.

## Vastgelegde workload

- Fysieke P6A Q8-bank: alle 241 projectierecords, gegroepeerd als
  `q/k/v/o/router/head`.
- Fysieke Q5-cache: experts 0–7 van alle 48 lagen; `gate_up` en `down` worden
  afzonderlijk afgestemd en daarna als één Q5-grafiek gemeten.
- Eén vaste FP32-activatievector uit NumPy `default_rng(120831)`.
- Geen H2D-kopieën in de getimede gebieden.
- CUDA-events op één non-blocking stream.

## Correctheidspoort

Iedere breedte in `{4,8,16,32,64}` moet voor de volledige Q8- en Q5-workload
eindige uitvoer leveren die bit-voor-bit identiek is aan de ongewijzigde
256-thread P6B-reductie. Een niet-exacte breedte wordt uitgesloten van selectie.
Ook de uiteindelijk gemengde Q8- en Q5-grafiek wordt opnieuw integraal met
ERVF-16 vergeleken.

## Validation en bevroren selectie

- Per projectietype en breedte: 3 warmups en 15 metingen.
- De meetvolgorde roteert per ronde en keert in oneven ronden om, zodat iedere
  breedte vroeg en laat in de thermische volgorde voorkomt.
- Per projectietype wordt onder de exacte varianten de laagste validation-p50
  gekozen. Bij een verschil kleiner dan 0,5% wint breedte 16 als die in de
  equivalentiegroep zit; anders wint de kleinste breedte. Dit is vooraf
  vastgelegd om ruisgedreven selectie te beperken.
- De gekozen grafieken worden daarna bevroren vóór de testreeks.

## Ongeopende test en primaire poorten

Baseline ERVF-16 en de bevroren kandidaat worden in 120 gepaarde metingen per
grafiek afwisselend als AB/BA uitgevoerd, na 10 warmups per variant.

Een bank slaagt alleen als:

1. alle individuele breedtes en de gemengde grafiek bitexact en eindig zijn;
2. `candidate_p50 / baseline_p50 <= 0,97`;
3. `candidate_p95 / baseline_p95 <= 0,97`.

Q8 en Q5 krijgen elk een zelfstandig verdict. `overall_pass` vereist dat beide
bankslagen. Een geïsoleerde pass is geen tok/s- of end-to-end-claim; die zou een
nieuwe, vooraf geregistreerde decoderreplicatie vereisen.

## Stopregels en bewijsgrens

CUDA-compilefouten, OOM, niet-eindige uitvoer of bitverschillen worden zonder
post-hoc kernelwijziging als fout/negatief resultaat bewaard. N1C bewijst of
falsificeert uitsluitend de lokale fysieke projectieplane-hypothese op deze
GPU en deze vaste modelgeometrie. Het is geen nieuwheidsclaim.
