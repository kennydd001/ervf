# N2B preregistratie — exact gecertificeerde LM-head search-oracle

Datum: 2026-08-12. Output ongeopend.

## Hypothese

Een vaste clustering van de 151.936 fysieke Q8-headrows kan met
centrum/radius-bovengrenzen minstens 60% van de rows overslaan en toch de
greedy argmax certificeren.

## Vaste clustering

- gebruik exact de gedequantiseerde Q8/BF16-scaleweights uit de fysieke P6-bank;
- tien orthonormale random hyperplanes, NumPy-seed `120820`, verdelen rows op
  hun tien tekenbits in maximaal 1.024 clusters;
- centrum = FP32 gemiddelde van iedere niet-lege cluster;
- radius = maximale Euclidische afstand tot dat centrum;
- bewaar per cluster ook maximale rownorm;
- geen clusteringparameter wordt na validation veranderd.

## Certificaat

Voor hiddenvector `h` is de reële upper bound
`h·c + ||h|| r`. Voeg conservatief FP32-product/reductiefout toe via
`gamma_4096 * ||h|| * (max_row_norm + ||c||)`, vergroot radius 0,1% voor de
radiusberekening en voeg één BF16-roundingmarge toe. Clusters worden op deze
bound verwerkt; een token is gecertificeerd wanneer de beste exact berekende
Q8-logit minstens alle resterende bounds is.

De bestaande runtime berekent voor de oracle nog alle logits; de oracle telt
hoeveel rows een toekomstige implementatie had moeten uitvoeren. Het is dus
geen timingclaim.

## Data en gates

- validation: alle 1.270 P0C-validationtokens; test opent alleen bij 100%
  certificatie, 100% dezelfde argmax en mediane row-skip `>=60%`;
- test: dezelfde poorten op de ongeopende P0C-testsplit;
- rapporteer mediane, p5 en minimale skipfractie en lege clusters.

Een validationfail sluit deze vaste Euclidische signcluster-variant en laat de
test dicht.
