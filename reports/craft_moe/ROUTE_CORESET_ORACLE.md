# H7 Route-Local Sparse Coreset Oracle — resultaat

Datum: 2026-08-10  
Status: compleet, `inconclusive_negative`  
Machineleesbaar resultaat: `reports/craft_moe/route_coreset_oracle.json`

## Hypothese en vooraf vastgelegde gate

H7 test of de originele, ongenormaliseerde top-6-routed som per token kan
worden gereconstrueerd met weinig reeds geselecteerde expertoutputs. Voor
`k=1..5` zijn alle subsets uitgeput en zijn vrije LS, exacte NNLS,
box-begrensde LS en vaste-routergewichtbaselines gemeten. De subsetkeuze
minimaliseerde uitsluitend routed-output-L2; de vocabulaire-KL is pas daarna
geopend.

De primaire NNLS-gate op validatie was:

- hogere empirische mediaan van minimaal `k` hoogstens 3, of p95 hoogstens 4,
  bij teacher→candidate-KL ≤`0,001`;
- falsificatie wanneer bij `k=5` meer dan 25% van de tokens KL >`0,003` houdt.

De testwindow was een vaste replicatie en is niet voor keuze of tuning
gebruikt. De preregistratie staat in
`reports/craft_moe/H7_ROUTE_CORESET_PREREGISTRATION.md`.

## Resultaat

| Split | mediaan min. k | p95 min. k | k≤3 | k≤4 | k=5, KL>0,003 | Oordeel |
|---|---:|---:|---:|---:|---:|---|
| Validatie, 256 tokens | 4 | 6 | 43,75% | 63,28% | 3,91% | gate faalt, niet gefalsificeerd |
| Test, 256 tokens | 4 | 6 | 41,41% | 55,47% | 2,34% | replicatie van gate-falen |

De gemiddelde minimumcardinaliteit bij KL ≤`0,001` is 3,590 op validatie en
3,723 op test. Dat correspondeert lokaal met respectievelijk 40,17% en 37,96%
minder actieve routed experts, maar de zware staart blijft: 18,75% van de
validatietokens en 21,88% van de testtokens heeft zelfs met de geselecteerde
`k=1..5`-kandidaten geen KL ≤`0,001` en valt daarom terug op exact top-6.

| NNLS | Validatie-KL | Test-KL | Validatie top-1 | Test top-1 |
|---:|---:|---:|---:|---:|
| k=3 | 0,003259 | 0,004142 | 96,88% | 96,09% |
| k=4 | 0,001406 | 0,001598 | 99,22% | 98,05% |
| k=5 | 0,000733 | 0,000698 | 98,83% | 99,61% |

Voor `k=5` is de 95%-sequence-block-bootstrap-KL
`[0,000576; 0,000890]` op validatie en `[0,000682; 0,000715]` op test. Vrije
LS en NNLS leveren dezelfde geaggregeerde curves; de gekozen vrije oplossingen
zijn bij `k=5` allemaal niet-negatief. Negatieve coëfficiënten zijn dus niet de
ontbrekende truc: de benodigde cardinaliteit zelf is de bottleneck.

## Controles

- De officiële-top-6-deltapatch geeft op beide splits exact KL `0`, top-1 `1`
  en CE-delta `0`.
- De handmatige BF16-reconstructie zonder deltapatch wijkt door de officiële
  reductieorde maximaal `2,0` af (`NRMSE 0,001005`). Deze numerieke ruis is niet
  als coresetfout meegeteld.
- Alle 44 tests slagen; de nieuwe suite bevat een bit-exacte original-control en
  een synthetisch probleem met bewezen tweearmige NNLS-oplossing.
- De volledige run duurde 9,58 s en gebruikte maximaal circa 1,18 GB CUDA-
  allocatie. Bronhash, commandoregel, gitstatus, hardware, coëfficiënten en alle
  tokenmetrics staan in het JSON-bestand.

## Stop/go

**Stop H7 na laag 26.** De vooraf vastgelegde positieve gate faalt, dus de
laag-23-interventie wordt niet geopend. H7 bewijst wel heterogene lokale
compressieruimte, maar niet de vereiste robuuste sparse-coresetstructuur en
zeker geen deploybare of gemeten snelheidswinst.

**Go naar de volgende onafhankelijke P0-oracle.** Er is niets op de testdata
herafgesteld; het negatieve H7-resultaat blijft append-only bewaard.

