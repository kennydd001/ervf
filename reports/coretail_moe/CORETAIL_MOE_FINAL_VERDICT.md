# CORETAIL-MoE - definitief masterverdict

**Einduitkomst: twee sterke technische deelbewijzen, maar geen inzetbare
Eureka. De geregistreerde totaalhypothese is op P2 gefalsificeerd.**

## Wat wel is bewezen

- De volledige bronbank bevat 6.144 echte GPTQ-experts en is onafhankelijk
  geverifieerd.
- CORETAIL reconstrueert alle 28.991.029.248 codes en alle 226.492.416
  BF16-schaalelementen bit-exact.
- Het werkelijk gebouwde full-bankformaat meet 1,993759 bpp en past volgens de
  gelockte residentformule in 7,725844 GiB tegenover 7,959961 GiB beschikbare
  VRAM. De full-bankaudit slaagt 26/26.
- De exacte fused microkernel is correct in 72/72 gevallen en haalt 33,319
  Gweight/s p50 en een conservatieve 30,738 Gweight/s p95-equivalent, boven de
  gate van 27,2 Gweight/s. De P1-audit slaagt 13/13.

Dit bewijst een echte **representatie- en kerneldoorbraak** voor deze bank:
een row-random-access ternary core plus sparse exacte tail kan fysiek compact,
exact en snel genoeg als geïsoleerde expertkernel worden uitgevoerd.

## Wat de totaalclaim breekt

De primaire full-depth kandidaat verhoogt de cross-entropy tegenover BF16 met
35,953% op validation en 42,943% op held-out test, bij slechts 61,811% test
top-1-overeenkomst. De preregistreerde grens was maximaal 2%. Alle vijf
domeinen falen; de onafhankelijke P2-audit slaagt 55/55.

De oorzaak ligt niet in CORETAILs exacte hercodering: P0 bewees dat deze exact
de al bestaande GPTQ-codes en schalen reproduceert. De kwaliteitsbreuk zit in
de onderliggende zeer lage-bit expertbank én de INT4-trunk; afzonderlijk meten
ze op test respectievelijk +23,762% en +11,087% relatieve CE, en samen +42,943%.

## Protocolconclusie

De testscore ligt boven 10%. Daarom:

- is de ene voorziene rank-8-repair niet geautoriseerd;
- is de geïntegreerde wall-clocktest niet geautoriseerd;
- mag de P1-projectie van 57,624 ms p50 / 62,262 ms p95 niet als end-to-end
  tokens-per-second worden geïnterpreteerd;
- is de deployment-Eureka **niet bereikt**.

De correcte wetenschappelijke uitkomst is dus niet "alles mislukt": het
fysieke formaat en de microkernel zijn overtuigend bewezen. Maar de combinatie
van geheugenfit, snelheid én kwaliteit die voor een inzetbare Eureka nodig is,
is in deze geregistreerde route weerlegd.
