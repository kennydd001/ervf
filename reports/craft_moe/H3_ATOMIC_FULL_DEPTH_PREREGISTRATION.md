# H3 Exact Atomic Expert Oracle — preregistratie gelijktijdig full-depth

Vastgelegd na de volledig positieve spreadmatrix en vóór full-depth-code of
inspectie van gelijktijdige uitkomsten. Dit is de eerste modelbrede
atom-oracleproef; exacte support blijft niet deploybaar.

## Vaste modelinterventie

- laag 0 blijft exact dense;
- in **alle 26 MoE-lagen 1–26** wordt tegelijk dezelfde fractie toegepast;
- acht vaste policy's: `{1.0,.75,.50,.35,.25,.15,.10,.05}`;
- per policy en laag volgt de kandidaat haar eigen officiële natuurlijke
  top-6-route en ongenormaliseerde routergewichten;
- support wordt op de eigen exacte kandidaat-activaties gekozen met stabiele
  globale `|p_e a_j| ||down_column_(e,j)||₂`-rangschikking;
- shared experts, attention, normen, residuals en alle overige gewichten
  blijven exact.

Op iedere MoE-laag wordt eerst de officiële volledige kandidaatlaag uitgevoerd
en vervolgens uitsluitend de routed bijdrage vervangen:

```text
candidate_next = BF16(official_full_candidate_next
                      + sparse_routed_candidate
                      - manual_full_routed_candidate)
```

De 100%-policy moet na iedere laag identiek blijven aan de teacher en finaal
KL `0`, CE-delta `0`, top-1 `1` geven.

## Vaste data

Dezelfde vier vooraf gedefinieerde 2×128-token-domeinen als de spreadfase:
WikiText-validatie, reeds geopende WikiText-test, drie lokale
instructieattachments en het lexicografische Pythoncorpus buiten alle
`craft_moe`-mappen. Bron-, concatenatie- en token-ID-hashes worden opnieuw
vastgelegd; geen corpus of fractionele keuze mag wijzigen.

## Raw bewijs en metrics

Voor iedere laag×policy×domein:

- eigen route-overlap met de teacher;
- lokale routed relatieve L2 en hidden-regressie;
- behouden atomcounts, ideale support-known BF16-bytes/MACs en tensor-lokale
  4-KiB-paginadruk;
- lossless bit-packed support.

Alle packed supports worden in één companion-safetensorsbestand geschreven,
met per tensor SHA-256 plus een bestandshash in de JSON. De verwachte omvang is
ruim onder 5 GiB. Finale metrics zijn volledige-vocabulaire KL/CE/top-1 met
10.000× gepaarde sequence-block-bootstrap, seed `20260810`, inclusief raw
per-tokenseries.

De evaluator gebruikt dichte nulmask-GEMM voor kwaliteit. Rekentijd is geen
sparse-runtime-, byte- of snelheidsmeting.

## Gates

De **primaire modelbrede 25%-gate** slaagt alleen wanneer op alle vier
domeinen:

1. relatieve CE-toename `<2%`;
2. gemiddelde teacher→candidate-KL `≤0,03`;
3. top-1-overeenkomst `≥90%`;
4. de 100%-control exact is.

De CE-moonshot is 10% met relatieve CE-toename `<3%` op alle vier domeinen en
wordt afzonderlijk gerapporteerd. Voor een veilige 10%-claim worden daarnaast
KL `≤0,05` en top-1 `≥85%` gerapporteerd, maar die extra criteria veranderen
de oorspronkelijke CE-moonshot niet achteraf.

Harde falsificatie: een 25%-domein met CE `≥2%`, KL `>0,10`, top-1 `<75%` of
een falende exact-control. Een uitkomst tussen primaire en harde grens is
inconclusief.

Alleen een positieve primaire full-depth-gate opent een vaste 1.024-token
candidate-validation en daarna een nieuw gepreregistreerd confirmatievenster.
Geen predictor, packed kernel, snelheid of definitieve Eureka-claim vóór die
replicatie. De reeds gefaalde tile-64-hardwaregate blijft ongewijzigd negatief.
