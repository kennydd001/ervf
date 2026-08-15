# H3 Exact Atomic Expert Oracle — preregistratie laag 23 downstream

Vastgelegd na de positieve, ongewijzigd bewaarde laag-26-screen en vóór code
of inspectie van atomaire laag-23-uitkomsten. Dit is de verplichte eerdere
interventie met exacte downstreamstaart; nog geen confirmatie of runtimeproef.

## Autorisatie en vaste hypothese

Laag 26 autoriseert deze stap omdat de vooraf gekozen globale bijdragescore
bij 25% en 10% retentie op validatie én test de CE-gates haalde. De selector
wordt niet opnieuw gekozen:

`score_(e,j) = |p_e a_(e,j)| ||down_column_(e,j)||₂`, met
`a_j=silu(gate_j(x))*up_j(x)`.

De hypothese is dat dezelfde exacte atom-support op laag 23 na officiële lagen
24–26 nog kwaliteit behoudt.

## Vaste data, kandidaten en uitvoering

- gepinde DeepSeek-V2-Lite- en WikiText-commits;
- eerste 256 validatie- en eerste 256 bestaande testtokens, gevormd als twee
  onafhankelijke sequentieblokken van 128 per split;
- officiële prefixlagen 0–22, interventie op laag 23, daarna officiële lagen
  24–26, final norm en volledige LM-head;
- natuurlijke top-6, originele ongenormaliseerde routergewichten; shared
  experts exact;
- globale stabiele topselectie over alle 8.448 routed atomen;
- vaste fracties `{1.0,.75,.50,.35,.25,.15,.10,.05}` met
  `ceil(f×8448)` atomen per token;
- ties volgen expert-slot- en neuronvolgorde;
- geen per-expert- of tegelselector en geen greedy-retuning in deze proef.

Kandidaten worden op de officiële laag-23-teacherstate geïnjecteerd als

```text
candidate23 = BF16(official_teacher23 + sparse_routed - manual_full_routed)
```

en vervolgens gezamenlijk maar onafhankelijk door exact dezelfde lagen 24–26
gevoerd. De 100%-control moet finale KL `0`, CE-delta `0`, top-1 `1` en lokale
relatieve L2 `0` geven.

## Metrics en accounting

Per fractie en split:

- lokale routed relatieve L2 op laag 23;
- hidden-NRMSE, router-top-6-overlap en routergewicht-NRMSE na lagen 24–26;
- finale volledige-vocabulaire teacher→candidate-KL, CE en top-1;
- gepaarde 10.000× sequence-block-bootstrapintervallen met seed `20260810`;
- lossless bit-packed support, behouden counts, ideale support-known
  BF16-weightbytes/MACs en tensor-lokale 4-KiB-paginadruk.

De evaluator mag dichte nul-gemaskerde GEMM gebruiken om kwaliteit te meten;
die tijd is geen sparse-runtime- of snelheidsmeting.

## Vooraf vastgelegde gates

Primaire downstreamgate bij 25% retentie:

1. relatieve CE-toename `<2%` op validatie én test (negatief telt als pass,
   niet als kwaliteitswinst);
2. gemiddelde finale KL `≤0,01` op beide splits;
3. finale top-1-overeenkomst `≥95%` op beide splits;
4. exacte 100%-control.

Alle vier openen pas de spread-layer/domainfase. De moonshotgate blijft 10%
retentie met relatieve CE-toename `<3%` op beide splits en wordt afzonderlijk
gerapporteerd; zij kan een mislukte primaire veiligheidsset niet overrulen.

Harde falsificatie treedt op als bij 25% op een split de relatieve CE-toename
`≥2%`, finale KL `>0,02` of top-1 `<90%` is. Een uitkomst tussen de
veiligheidsgate en harde grens is inconclusief. Bootstrapintervallen zijn
verplicht gerapporteerd maar wegens slechts twee blokken niet zelf gatevormend.

Geen predictor, atomic index, packed kernel of Eureka-claim vóór een positieve
downstreamuitkomst.
