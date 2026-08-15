# Definitief onderzoeksverdict - BITFLOW, offload-roofline en CORETAIL

**Eindantwoord: de onderzochte routes zijn afgewerkt, maar een volledige
Eureka is niet bewezen. CORETAIL levert wel een nieuw en stevig
representatie- plus microkernelresultaat.**

| Onderzoekslijn | Definitieve status | Sterkste bewijs |
|---|---|---|
| BITFLOW-MoE | Lineaire tak negatief en gesloten | Validation/test CE-damage recovery -645,95% / -395,02%; onafhankelijke audit 23/23 |
| Offload-roofline | Gemengde deelresultaten, geen Eureka | LFU en permutatie negatief; lokale H2D-roofline geverifieerd maar K3-claim conditioneel; oorspronkelijke Q4-runtime niet uitgevoerd |
| CORETAIL-MoE P0 | Geslaagd | Exact full-bankformaat, 1,993759 bpp, 7,725844 GiB residentformule; audit 26/26 |
| CORETAIL-MoE P1 | Geslaagd | Exacte kernel 30,738 Gweight/s conservatief versus gate 27,2; 72/72 correct; audit 13/13 |
| CORETAIL-MoE P2 | Negatief en gesloten | Primaire relatieve CE +35,953% validation en +42,943% test versus maximaal 2%; audit 55/55 |

## Ontdekking

De universele ternary core plus sparse exacte tail is op deze complete
Qwen-bank niet langer alleen een algebraïsch idee. Het formaat is werkelijk
gebouwd, fysiek geteld, bit-exact gedecodeerd en door een eigen fused kernel
boven de geregistreerde throughputgate uitgevoerd. Dat is het positieve,
herbruikbare onderzoeksresultaat.

## Waarom dit nog geen inzetbare Eureka is

Een deployable Eureka vereiste tegelijk geheugenfit, correcte en voldoende
snelle uitvoering, modelkwaliteit en daarna een geïntegreerde wall-clockpass.
CORETAIL passeert de eerste twee, maar faalt de full-depth kwaliteit zeer ruim.
De held-out testscore van +42,943% relatieve cross-entropy ligt boven de harde
10%-stopgrens. Daarom zijn repair en wall-clock volgens het vooraf vastgelegde
protocol gesloten. Een microkernelprojectie mag niet tot tok/s worden
opgewaardeerd.

## Wetenschappelijk eindpunt

- **Bewezen:** complete exacte compacte representatie en geïsoleerde
  microkernelhaalbaarheid voor deze GPTQ-bank.
- **Weerlegd:** dat deze exacte GPTQ-experts met een INT4-trunk binnen de
  geregistreerde kwaliteitsgrens inzetbaar zijn.
- **Niet beweerd:** end-to-end tokens per seconde, brede generalisatie naar
  andere MoE-families of een geslaagde Q4 async-cache-runtime.

Dit sluit de drie aangevraagde onderzoekspakketten zonder open experimentele
claim. Een volgende route moet met een aantoonbaar veel sterkere
kwaliteitsbasis beginnen; verdere systeemoptimalisatie van deze kandidaat kan
de gemeten kwaliteitsfout niet oplossen.

De finale onafhankelijke P2-audit slaagt 55/55 en de volledige repositorysuite
156/156.
