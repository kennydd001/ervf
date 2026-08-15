# FLEQ-MoE P1 — expert-streamed Qwen GSQ-smoke

**Vooraf geregistreerd:** 2026-08-11  
**Status:** geen FLEQ-evaluatiemetrics geopend of berekend.

## Vraag

Kan de gepinde officiële GSQ-codebookoperator op deze Windows/Blackwell-stack
één echte Qwen3-MoE-expert tegelijk optimaliseren, binnen 7,5 GiB VRAM, en op
ongeziene activaties aantoonbaar beter zijn dan zijn eigen 2-bit of ternary
initialisatie?

Dit is een infrastructuur- en lokaal-oraclescherm. Een positieve uitkomst is
geen bewijs voor full-depth CE, rollouts, artifactomvang of snelheid.

## Bevroren inputs

- Model: `Qwen/Qwen3-30B-A3B-Base`.
- Revisie: `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
- Lagen: `[0, 47]`, zodat een vroege en late MoE-laag worden geraakt.
- Activatiebron: uitsluitend de bestaande P1C-**validation**capture met SHA-256
  `5e87ba7cd694cb2c297b1b54e8ea48e4eee665a3ad573a557c6a7da9e88a2622`.
- Context 0 is calibratie/training; context 1 is held-out smoke-evaluatie.
- De bestaande P1C-testcapture wordt niet gebruikt.
- Per laag worden de acht experts met de meeste context-0-invocaties gekozen;
  ties worden op oplopend expert-ID gebroken. Deze regel staat vast vóór de
  counts worden geopend.
- GSQ-codecommit: `03fc16484c369e3127225615d5e03e8d3a6043e3`.

## Kandidaten

1. Symmetrische groupwise 2-bit RTN, groupsize 128.
2. Symmetrische groupwise 2-bit GPTQ, groupsize 128.
3. GSQ 2-bit vanaf de GPTQ-initialisatie.
4. Ternary RTN/GSQ wordt alleen binnen dezelfde smoke toegevoegd wanneer de
   gepinde ternary operator technisch draait zonder de 2-bitdata of gates te
   wijzigen.

Voor GSQ blijven de officiële codebookklassen, BF16 logits, codewaarden,
Gumbeloperator, tien epochs, temperatuur `2,0→0,05`, logit scale `100→500`,
Lion-betas `[0,9, 0,95]`, assignment-LR `1e-4`, scale-LR `5e-5` en seed
`260811` vast. Alleen de quantizers worden expert-per-expert opgebouwd om het
geheugenplafond te respecteren.

## Metingen

Voor iedere laag/expert/methode:

- calibratie- en held-out relatieve L2-fout van de volledige SwiGLU-expertoutput;
- held-out cosine-overeenkomst en router-massgewogen fout;
- gate/up/down gewichtsfout en werkelijke codehistogrammen;
- effectieve codebits plus analytische scale-overhead;
- peak allocated/reserved CUDA-bytes, proces-RSS en wandtijd;
- determinisme-herhaling op één vooraf gekozen expert per laag;
- volledige BF16-fallbackidentiteit als controle.

## Gates

De smoke is `infrastructure_positive` wanneer tegelijk:

1. alle outputs, losses, gradients, codes en scales eindig zijn;
2. peak CUDA allocation maximaal 7,5 GiB en proces-RSS maximaal 32 GiB is;
3. de BF16-referentie/fallback exact is;
4. GSQ 2-bit op held-out data de router-massgewogen expertoutput-MSE minstens
   20% verlaagt tegenover GPTQ 2-bit;
5. minstens zes van acht geselecteerde experts per laag verbeteren en geen
   laag een p95-foutregressie boven GPTQ vertoont;
6. de determinismecontrole binnen `1e-6` relatieve metriekafwijking sluit.

De smoke is `software_blocked` uitsluitend bij een reproduceerbare
platform-/dependencyfout vóór een geldige optimalisatie. Numeriek slechte
kwaliteit is `smoke_negative`, niet blocked.

## Escalatiegrens

Alleen `infrastructure_positive` staat een afzonderlijk vooraf geregistreerde
drie-laags-PTQ-test toe. P1 mag geen full-modelartifact, test-CE, benchmark,
512-tokenrollout, packed-runtime of Eureka claimen.

