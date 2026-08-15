# H6 Co-routed Quantization Error Cancellation

## Definitief oordeel

**H6 is in de vooraf vastgelegde fase A gefalsificeerd.** De globale
cancellation fraction van de zes gewogen Q3-expertfouten is `-1,129%` op
validatie en `-0,106%` op test. De absolute waarde is daarmee op beide splits
kleiner dan de vooraf geregistreerde `2%`-grens voor een near-zero kruisterm.
Het minteken betekent bovendien dat co-routing de totale fout in deze meting
licht versterkt in plaats van annuleert.

Volgens de preregistratie is dit een harde stop. Er zijn daarom geen gains,
clippinggrenzen of afrondingsvarianten op de data gefit en er is geen
laag-/domeinspread of full-depth-proef geopend. Dit voorkomt dat een negatief
mechanistisch resultaat achteraf wordt vervangen door een vrijere optimizer.

## Fase-A-decompositie

Voor iedere routeslot is de foutvector exact gedefinieerd als
`v_s = p_s * (Q3_expert_s(x) - BF16_expert_s(x))`. De primaire ratio gebruikt
de sommen over alle tokens:

`cancellation = (sum diagonal - sum aggregate) / sum diagonal`, met
`diagonal = sum_s ||v_s||^2` en `aggregate = ||sum_s v_s||^2`.

| Metric | Validatie | Test | Harde stop |
|---|---:|---:|---:|
| tokens | 256 | 256 | — |
| diagonale energiesom | 41.490,371 | 280.038,173 | — |
| aggregate energiesom | 41.958,631 | 280.335,845 | — |
| kruistermsom | +468,260 | +297,672 | — |
| cancellation fraction | **−1,129%** | **−0,106%** | `abs <2%` op beide |
| near-zero | ja | ja | **getriggerd** |

De veel grotere testenergiesom verandert de conclusie niet: de beslissing is
een ratio van energiesommen per split en beide onafhankelijke splits sluiten
dezelfde near-zero-preconditie. De ruwe tokenwaarden en slotcomponenten zijn
behouden, zodat de ratio zonder modelherberekening kan worden gecontroleerd.

## Controls en same-byte-accounting

- de officiële teacher-delta-control is bitexact;
- opnieuw berekende routergewichten hebben maximale absolute fout `0,0` en de
  zes expert-ID's plus slotvolgorde zijn exact gelijk;
- alle 64 routed experts zijn in de capture geraakt;
- de bestaande opgeslagen Q3-output is, zoals vooraf bepaald, alleen een
  batchvormregressiediagnostiek: NRMSE `0,003467`, niet een bitexacte control;
- de hypothetische per-row gain zou een bestaande down-dequantisatieschaal
  vervangen: 131.072 bestaande en kandidaatwaarden, 262.144 bytes in beide
  gevallen, dus nul extra waarden/bytes en geen nieuwe kerneloperand.

Dat laatste is alleen een analytische layoutcontrol. Omdat fase B niet mocht
openen, is er geen kandidaatquantizer en dus ook geen fysieke Q3-kernel- of
runtimeclaim.

## Reproduceerbaarheid en artefacten

De covariance-run gebruikte exact DeepSeek-V2-Lite revision
`604d5664dddd88a0433dbae533b7fe9472482de0` en WikiText-2 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. De gemeten rekentijd voor capture,
decompositie en schrijven was `1,385 s`; dit is geen modelruntimebenchmark.

- beslissende adjudicatie: `qerc.json`, 35.434 bytes, SHA-256
  `681284583baf0b08d39dd5c153e184b008f514d16ecf86c543c520816db42cc3`;
- ruwe covariance: `qerc_covariance_layer26.json`, 108.293 bytes, SHA-256
  `e26d802a68b7f106dd5157d623eabbdcbd1eade26725408f2f8c6b86def90946`;
- lossless componentartifact: `qerc_layer26_components.safetensors`,
  31.480.824 bytes, SHA-256
  `4ecb801589b6221567edaaab390ea2373665708d4ad815f83e15a0feedfcc1f0`.

De repository heeft nog geen commit (`revision: null`) en is dirty; dat staat
expliciet in de JSON. Bibliotheekversies, volledige commands, cwd en inputhashes
zijn eveneens opgeslagen.

## Reikwijdte van de falsificatie

Het resultaat falsificeert de geregistreerde natuurlijke co-route-
foutcancellatie als noodzakelijke basis voor QERC. Het bewijst niet dat iedere
denkbare gezamenlijke quantizer onmogelijk is. In het bijzonder zijn vrije
integercodewijzigingen, extra metadata, andere bitbreedtes of een generiek
supervised gainmodel andere hypotheses met andere kosten. Ze mogen deze
gesloten H6-gate niet vervangen.

**Stop H6:** geen fase-B-fit, roundingonderzoek, spread of full-depth.
**Go onderzoeksprogramma:** vervolg met de onafhankelijke P1-hypothese H8
Cache Span Minimization.
