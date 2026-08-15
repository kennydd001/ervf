# CRAFT-MoE masterverdict

## Uitkomst

**De geregistreerde CRAFT-MoE-hypothesefamilie is overtuigend gesloten zonder Eureka.** Geen enkele kandidaat voldoet aan de gezamenlijke V2-gate. Alle niet-geblokkeerde hypothesen hebben een terminal negatief besluit; de geblokkeerde engineeringstappen zijn volgens de vooraf vastgelegde stopregels terecht niet uitgevoerd.

Dit is geen universeel onmogelijkheidsbewijs voor MoE-compressie. Het is wel voldoende bewijs om deze specifieke route–bit–atom–cache-stack niet verder post-hoc te variëren en niet naar V4 Flash te escaleren.

## Gemeten feiten

- **Mass-Budget is een bevestigde incrementele baseline, geen CRAFT-Eureka.** Op het vooraf vastgelegde testvenster van 2.048 tokens reduceert delta=0,004 de expertloads met 14,017%, bij KL 0,003704 en relatieve CE -0,057% (95%-blokinterval -0,171% tot +0,060%); alle 16 blokken besparen loads.
- **Alle niet-geblokkeerde CRAFT-hypothesen bereikten een terminaal negatief besluit.** H1, H2, H3, H4, H6 en H10 zijn gefalsificeerd; H7 en H8 zijn inconclusief-negatieve screens met gefaalde positieve gates; H5, H9 en packed runtime stopten door afhankelijkheden.
- **Er bestaat geen full-depth inzetbare CRAFT-kandidaat.** Lokale H1/H3-oraclewinsten overleefden de vereiste eerdere-laag- of gelijktijdige-full-depthtest niet; daarna was geen predictor of kernel toegestaan.
- **Het technische verdict is onafhankelijk reproduceerbaar.** De bevroren reproduceerbaarheidsaudit bevat 751 oorspronkelijke controles: 748 geslaagd, nul verplichte fouten, drie claimbegrenzende waarschuwingen en een hashmanifest met 136 entries.
- **Brede nieuwheid wordt niet ondersteund.** De novelty-audit omvat acht verplichte families, vijf claim-eenheden en 34 primaire/official/patentbronnen; geen brede claim blijft overeind.

## Technische hypothesen

| ID | Terminale status | Doorslaggevend resultaat |
|---|---|---|
| H7_ROUTE_CORESET | `inconclusive_negative` | Validatie/test minimum-k-mediaan 4 en empirische p95 6; de positieve mediaan<=3- of p95<=4-gate faalde. |
| H1_CRCQ | `falsified_downstream` | De gezamenlijke laag-26-oracle vraagt 9,831%/12,240% upgrades, maar de laag-23 exact-tailinterventie 75,521%/70,508%. |
| H3_ATOMIC_ORACLE | `falsified_full_depth` | De lokale/spread-oracles bij 25% en 10% waren positief; gelijktijdig full-depth 25% faalt op +2,1129% WikiText-test-CE en instructie-KL 0,03505. |
| H4_SKETCHGATE | `falsified` | KL-recovery slaagt, maar high-damage false negatives, attributie en hardwaremodelgates falen. |
| H2_BLOCK_COALESCING | `hard_falsified` | Alle 1.280 exacte ILP's zijn optimaal; de block-8 natural-unionreductie is slechts 19,65%/20,24%, zelfs onder de 25%-hardstop. |
| H5_ATOMIC_INDEX | `blocked_by_H3` | De predictor was verboden nadat de gelijktijdige full-depth atomic-gate faalde. |
| H6_QERC | `hard_falsified_phase_a` | Natuurlijke Q3-cancellation is -1,129% validatie en -0,106% test; beide liggen in de vooraf vastgelegde near-zero-stopband. |
| H8_CACHE_SPAN | `inconclusive_negative` | Oracle-missreductie is 41,35%/48,54% en span-uplift boven zero-fill slechts +1,442/+0,225 procentpunt. |
| H9_BISPARSE | `blocked_by_H3` | Het bi-sparse-kernelpad was verboden nadat de gelijktijdige full-depth atomic-gate faalde. |
| H10_REDUCTION_ORDER | `hard_falsified` | Held-out Q3→Q4-KL-gapclosure is 1,487%/0,829% tegenover de >=20%-gate en <10%-hardstop. |
| PACKED_RUNTIME | `blocked` | Geen methode haalde de vereiste geprojecteerde winst en kwaliteits-/downstreamgates. |

## Revolutionaire V2-gate

De gate geldt conjunctief: één en dezelfde kandidaat moet alle voorwaarden halen. De uitkomst is 0 van 6 bewezen voorwaarden.

| Gate | Eis | Status | Bewijs |
|---|---|---|---|
| G1_ACTIVE_EXPERT_BYTES | >=4,0× minder actieve expertbytes dan packed int4 + Mass-Budget delta=0,004 | `not_demonstrated` | Er bestaat geen full-depth, kwaliteitsgekwalificeerde CRAFT-kandidaat of packed layout. De 25%-BF16-atomic-oracle is ideaal 4 effectieve bits per volledig expert: slechts gelijk aan gewone int4 vóór metadata en vóór Mass-Budgets loadbesparing. |
| G2_WALLCLOCK_DECODE | >=2,0× gemeten batch-1-decodespeedup versus packed int4 + Mass-Budget | `not_evaluated_dependency_stop` | PACKED_RUNTIME was dependency-geblokkeerd. De H4/H8-CUDA-metingen zijn componentmicrobenchmarks zonder temperatuur-/kloktelemetrie en expliciet geen runtimeclaims. |
| G3_RELATIVE_CE | <2% relatieve cross-entropyschade voor dezelfde kandidaat | `not_satisfied_by_any_byte_qualified_candidate` | De sterkste relevante full-depth H3-25%-oracle bereikt +2,1129% WikiText-test-CE en faalt ook lokale-instructie-KL (0,03505 > 0,03). Mass-Budget slaagt voor zijn eigen CE-gate, maar benadert de vereiste bytereductie niet. |
| G4_LONG_ROLLOUTS | >=512 gegenereerde tokens op >=20 prompts over meerdere taaktypes | `not_evaluated_no_candidate` | Geen CRAFT-methode slaagde voor de vereiste oracle-, downstream- en full-depth-gates. Korte oudere Mass-Budget-smokerollouts voldoen niet aan deze gate. |
| G5_P95_LATENCY | Geen problematische p95 batch-1-decodelatency | `not_evaluated_no_packed_runtime` | Er is geen packed end-to-end kandidaat-runtime of p95-verdeling. |
| G6_SECOND_MODEL | Alle relevante gates repliceren op een tweede MoE-familie | `not_evaluated_v2_gate_failed` | De onveranderlijke stopregel verbiedt V4 Flash of escalatie naar een tweede familie voordat DeepSeek-V2-Lite de volledige revolutionaire gate haalt. |

## Afgeleide boekhouding

- **Geprojecteerde strict-I/O-ratio van Mass-Budget:** 201,48 / 173,24 = 1,163011× (14,0163% geprojecteerde bytebesparing). Grens: Alleen deterministische packed-int4-boekhouding; geen latency, throughput, energie of gemeten transfertraffic.
- **H2-test-block-unionfactor:** 1 - 0,2024 = 0,7976, gelijk aan slechts 1,2538× minder unieke-experteenheden in de geïdealiseerde uniontelling. Grens: Eén exacte late-laag-oracle; geen speculative runtime- of full-depthclaim.
- **H1 effectieve lokale precisie:** Laag 26: 3 + 0,12240 = 3,12240 bits; laag-23-test: 3 + 0,70508 = 3,70508 bits. Grens: Oracle-geselecteerde masks vereisen teacherwerk en vormen geen causale runtimecontroller.
- **Ideale BF16-atomfractie versus int4:** 25% × 16 bit = 4,0 effectieve bits; 10% × 16 bit = 1,6 effectieve bits, of maximaal 2,5× versus int4 bij gelijke loads. Grens: Negeert indices, tiles, cachelines en kernels. Beide gelijktijdige full-depthpolicies faalden; 25% is op ruwe weightbits niet beter dan gewone int4.
- **Geen geldige multiplicatieve CRAFT-stack:** Route-, bit-, atom- en cachefactoren mogen niet worden vermenigvuldigd, omdat ze op verschillende interventies zijn gemeten en meerdere vereiste gates faalden. Grens: Alle eerdere 7,36/4,94-GB-per-output-stackramingen blijven hypothetische scenario-aritmetiek, geen resultaat.

## Subjectieve inferentie

- Het dominante obstakel is cumulatief full-depthgedrag en uitvoerbare databeweging, niet een ontbrekende kleine predictortweak.
- Mass-Budget blijft een nuttige incrementele baseline, maar zijn 14% geprojecteerde loadbesparing verschilt kwalitatief van het >=4×-boven-int4-doel.
- Verdere post-hocvarianten binnen deze hypothesefamilie zouden het vooraf vastgelegde bewijs verzwakken, niet een geloofwaardige doorbraak creëren.
- De sterkste onderzoeksbijdrage is de negatieve kaart: exacte oracleplafonds, dependencystops en onafhankelijke gateverificatie.

## Novelty-status

De onafhankelijke audit vindt **geen verdedigbare brede nieuwheidsclaim**. CU1 en CU3 zijn hoogstens `possibly novel intersection`: Beide exacte doorsneden werden slechts niet gevonden in een begrensde search en beide zijn technisch gefalsificeerd; dit ondersteunt geen nieuwheids- of praktische claim. De gezamenlijke optimizer is `not searched sufficiently` en niet geïmplementeerd; de kernel/layoutclaim is breed `clearly prior art`; geen CRAFT-implementatie.

## Exacte volgende actie

1. Bevries dit CRAFT-pakket en markeer de registry `closed_no_eureka`.
2. Download of test DeepSeek V4 Flash niet vanuit deze onderzoekslijn.
3. Bouw geen packed kernel voor de gefaalde kandidaten en claim geen snelheid uit projected bytes of componentmicrobenchmarks.
4. Start alleen opnieuw via een nieuw registry-item met een mechanistisch onafhankelijke hypothese en vooraf vastgelegde oracle-, full-depth-, runtime- en tweede-modelgates.

## Beperkingen

- Dit sluit het geregistreerde CRAFT-MoE-hypothesepakket, niet iedere denkbare MoE-compressiemethode.
- H7 en H8 zijn inconclusief-negatief, geen universele onmogelijkheidsbewijzen.
- Er is geen fysieke packed-runtimebaseline gebouwd, omdat geen afhankelijke kandidaat de stopgates haalde.
- De lokale repository heeft geen Git-commit; integriteit rust op gepinde upstreamrevisions en artefacthashes.
- De novelty- en patentsearches zijn begrensd en ondersteunen geen juridische conclusie.

Eindstatus: `closed_no_eureka`. Sluitingsgrond: de onderzochte hypotheses zijn overtuigend gefalsificeerd of volgens preregistratie dependency-geblokkeerd.
