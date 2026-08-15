# CRAFT-MoE novelty-auditprotocol

Vastgelegd op 2026-08-10 vóór het schrijven van de claimmatrix en het
nieuwheidsverdict. Deze audit mag technische gates niet wijzigen, negatieve
experimenten niet promoveren en een niet-gevonden bron niet als bewijs van
nieuwheid behandelen.

## Scope en bronbeleid

De audit splitst CRAFT-MoE in vijf afzonderlijke claim-eenheden:

1. gezamenlijke selectie van alternatieve route en quantisatiebits;
2. blockbrede routecoalescing over expliciete equivalentieklassen;
3. randomized residual syndrome voor precisieacquisitie;
4. een gezamenlijke route–bit–atom–cache-optimizer;
5. een custom kernel- of layoutclaim.

Voor technische beweringen gelden primaire papers, officiële proceedings,
officiële code en gepubliceerde patentdocumenten als bewijs. Secundaire pagina's
worden hoogstens gebruikt om een primaire bron te vinden. De zoekcutoff is
2026-08-10. De patentcontrole is een beperkte trefwoordcontrole in Google
Patents en geen juridisch onderzoek, claim chart, freedom-to-operate-analyse of
uitspraak over patentability.

## Verplichte vergelijkingsfamilies

- Cache-Conditional Experts / Max Rank / Cache-Prior;
- Counterfactual Routing Analysis;
- BuddyMoE, SERE, MoE-ERAS en ReMoE;
- D²MoE, SliceMoE, MoBiQuant en routing-consistent PTQ;
- FloE, intra-expert activation sparsity, MoE-Prism en Mixture of Neuron Experts;
- EcoSpec, AcceptMoE, EdgeXpert, MoE-Spec en speculative expert prefetch;
- ZEDA, BEAM en post-trained dynamic expert skipping;
- random/JL sketches en adaptive bitplane acquisition.

## Beslisregel

Iedere claim-eenheid krijgt exact één van deze labels:

- `clearly prior art`;
- `close/overlapping`;
- `possibly novel intersection`;
- `not searched sufficiently`.

`possibly novel intersection` betekent uitsluitend dat de exacte doorsnede in
de gerichte search niet is gevonden. Het is geen nieuwheidsbewijs. Een
technisch gefalsificeerde of nooit geïmplementeerde claim kan niet door de
literatuuraudit worden opgewaardeerd tot een Eureka-uitkomst.

De audit is compleet wanneer alle acht families en vijf claim-eenheden in een
machineleesbare matrix staan, alle bron-URL's uniek en controleerbaar zijn, het
labelvocabulaire exact sluit en het rapport de zoekbeperkingen en technische
status expliciet vermeldt.
