# H3 Exact Atomic Expert Oracle — laag 26

## Oordeel

**De vooraf geregistreerde laag-26-oracle is positief, inclusief de
moonshotgate.** De globale exacte bijdragescore behoudt 25% van de 8.448
routed atomen per token met een relatieve CE-delta van `+0,1118%` op validatie
en `−0,1390%` op test. Bij 10% retentie is dit `+0,4086%` en `+0,0079%`.
Beide liggen ruim onder respectievelijk de vaste 2%- en 3%-grens.

Dit is nog geen deploybare of modelbrede Eureka. De support wordt uit de
exacte SwiGLU-activaties afgeleid, alleen de laatste MoE-laag is ingegrepen en
de tile-64-hardwaregate faalt. De uitkomst opent daarom laag 23 plus de exacte
downstreamstaart; zij opent nog geen predictor of snelheidsclaim.

## Hypothese en vaste proef

Voor ieder van de zes natuurlijke routed experts is de output exact ontbonden
als

`a_j = silu(gate_j(x))*up_j(x)` en
`v_(e,j)=p_e*a_(e,j)*down_column_(e,j)`.

De shared experts bleven exact. Kandidaten zijn op de officiële teacherstate
geïnjecteerd als
`BF16(teacher + sparse_routed − manual_full_routed)`. Vooraf stonden 256
validatie- en 256 bestaande testtokens, zes selectors en de retentiecurve
`{100,75,50,35,25,15,10,5}%` vast. De primaire selector was globale
`|p_e a_j| ||d_j||₂`; testdata is niet gebruikt om de selector of fractie te
wijzigen.

## Primaire curve

| Retentie | Val KL | Val rel. CE | Val top-1 | Test KL | Test rel. CE | Test top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 0 | 0,000% | 100,00% | 0 | 0,000% | 100,00% |
| 75% | 0,000098 | +0,0319% | 100,00% | 0,000091 | +0,0166% | 100,00% |
| 50% | 0,000389 | +0,0225% | 99,22% | 0,000438 | +0,0819% | 99,61% |
| 35% | 0,000724 | +0,0907% | 99,22% | 0,000797 | +0,0240% | 99,22% |
| **25%** | **0,001106** | **+0,1118%** | **98,44%** | **0,001425** | **−0,1390%** | **98,44%** |
| 15% | 0,002380 | +0,4648% | 98,05% | 0,003369 | +0,2150% | 97,27% |
| **10%** | **0,003954** | **+0,4086%** | **97,27%** | **0,005716** | **+0,0079%** | **96,48%** |
| 5% | 0,007167 | −0,1172% | 96,48% | 0,011224 | +0,4630% | 94,14% |

De gepaarde sequence-block-bootstrap voor relatieve CE is bij 25% retentie
`+0,0288%–+0,1997%` op validatie en `−0,2437%–−0,0486%` op test. Bij 10% is
dit `+0,2255%–+0,5816%` en `−0,1181%–+0,1167%`. Er zijn per split slechts twee
blokken; deze intervallen zijn dus exploratief. Een negatieve CE-delta wordt
niet als kwaliteitswinst geïnterpreteerd.

## Selector- en hardwarediagnostiek bij 25%

| Selector | Val KL | Test KL | Val rel. CE | Test rel. CE |
|---|---:|---:|---:|---:|
| per expert `|a|` | 0,002353 | 0,003577 | +0,1244% | +0,0924% |
| per expert `|a| ||d||₂` | 0,002498 | 0,003638 | +0,2850% | +0,0016% |
| **globaal atomair** | **0,001106** | **0,001425** | **+0,1118%** | **−0,1390%** |
| tile 16 | 0,006467 | 0,007977 | −0,3317% | +0,0194% |
| tile 32 | 0,007546 | 0,010233 | −0,1333% | −0,0814% |
| tile 64 | 0,009547 | 0,012085 | −0,3521% | −0,0913% |

Tile 64 heeft bij 25% **8,63×** de globale neuron-KL op validatie en **8,48×**
op test, tegenover de vaste bovengrens `1,20×`. De hardwaregate faalt dus
ondubbelzinnig, ondanks toevallig gunstige CE-puntschattingen.

Ideale support-known BF16-accounting daalt lineair tot 25% van de routed
weightbytes en MACs. In de bestaande rij-geordende tensors is de analytische
4-KiB-paginadruk voor de globale selector echter nog 50%: ieder gekozen
down-kolomelementpatroon doorkruist alle pagina's van die expertmatrix. Tile 64
verlaagt die tensor-lokale paginafractie tot circa 35,3%, maar verliest veel te
veel KL. Dit zijn analytische cijfers; de evaluator gebruikte dichte GEMM's met
nulmaskers en mat geen sparse runtime.

## Controles en reproduceerbaarheid

- de trace-route-ID's en routergewichten zijn exact gereproduceerd;
- de 100%-deltacontrol heeft op ieder token KL `0`, CE-delta `0`, top-1 `1` en
  lokale relatieve L2 `0`;
- een apart gelanceerde directe BF16-GEMM is niet bitexact door sporadische
  één-stapsafronding: NRMSE `4,42×10⁻⁶`, maximum `0,00390625`; dit staat rauw in
  het resultaat en beïnvloedt de exact-control niet;
- alle 43 supports per split zijn lossless bit-packed met SHA-256 opgeslagen;
- 10.000 bootstrapresamples, seed `20260810`; repository had nog geen commit en
  was dirty;
- ruwe JSON: `reports/craft_moe/atomic_oracle.json`, SHA-256
  `8af20192684b5427b293a858a90b711a9e1d364c85daa6395058bb1acd592bb9`.

## Beperkingen en stop/go

De uitkomst bewijst dat de late routed experts per token een verrassend dunne
exacte atom-support hebben. Zij bewijst niet dat die support vóór gate/up
goedkoop voorspelbaar is, dat dezelfde fracties vóór laag 26 downstream
overleven, dat code/instructie hetzelfde patroon hebben, of dat bytes/latency
werkelijk dalen.

**Go:** preregistreer laag 23 met exact uitgevoerde lagen 24–26 en alleen de
reeds gekozen globale selector op 25% en 10% als primaire kandidaten. Daarna
pas spread-layers/domeinen en exact-greedy. **Stop:** geen predictor, packed
kernel of Eureka-claim zolang de downstream- en hardwareproblemen niet zijn
opgelost.
