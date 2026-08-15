# H3 Exact Atomic Expert Oracle — preregistratie spread-layers en domeinen

Vastgelegd na de positieve laag-23-downstreamproef en vóór spread-code of
inspectie van spread-uitkomsten. Deze fase test laag- en domeingevoeligheid met
exact dezelfde selector; zij is nog geen gelijktijdige full-depth-interventie.

## Vaste selector en lagen

De enige selector blijft

`score_(e,j)=|p_e a_(e,j)| ||down_column_(e,j)||₂`.

Per token wordt stabiel globaal over de 8.448 routed atomen gerangschikt. De
fracties blijven `{1.0,.75,.50,.35,.25,.15,.10,.05}` met `ceil(f×8448)`
atomen. Shared experts blijven exact; natuurlijke top-6-gewichten worden niet
hergenormaliseerd.

De interventielagen zijn exact `{1,13,26}`. Iedere laag wordt afzonderlijk
getest met officiële prefix, routed-deltapatch op die laag en alle resterende
officiële decoderlagen tot en met 26. Er wordt in deze fase dus nooit meer dan
één laag tegelijk gesparsificeerd.

## Vaste corpora

Iedere cel gebruikt twee sequentieblokken van 128 tokens:

1. `wikitext_validation`: eerste 256 tokens van de gepinde validatiesplit;
2. `wikitext_test`: eerste 256 tokens van de reeds geopende testsplit;
3. `local_instruction`: eerste 256 tokens van de concatenatie, in opgegeven
   volgorde, van de drie user-supplied attachments met IDs
   `35a0d2b1...`, `7bdae3b8...`, `49e95e1b...`;
4. `local_code`: eerste 256 tokens van lexicografisch geconcateneerde `*.py`-
   bestanden onder `scripts`, `src` en `tests`, met uitsluiting van
   `scripts/craft_moe`, `src/moe_lab/craft_moe` en `tests/craft_moe` zodat deze
   onderzoeksfase haar eigen corpus niet verandert.

Bestandspaden en SHA-256's worden vóór tokenisatie in de resultaat-JSON
opgeslagen. De lokale corpora zijn transferchecks, geen onafhankelijke
confirmatieset. WikiText-test is niet opnieuw geheim en mag niets tunen.

## Vaste uitvoering en metrics

Per laag wordt voor alle vier domeinen in één vaste batch de officiële teacher
uitgevoerd. Kandidaten gebruiken

`BF16(official_teacher_L + sparse_routed_L - manual_full_routed_L)`

en lopen daarna exact full-depth. Per laag×domein×fractie worden lokale routed
relatieve L2, finale volledige-vocabulaire KL/CE/top-1, 10.000× gepaarde
sequence-block-bootstrap, lossless support, counts, ideale BF16-bytes/MACs en
tensor-lokale 4-KiB-pagina's opgeslagen. Na iedere downstreamlaag worden voor
25% en 10% minimaal hidden-NRMSE en router-top-6-overlap gerapporteerd.

De 100%-control moet voor iedere laag en ieder domein finale KL `0`, CE-delta
`0`, top-1 `1` en lokale relatieve L2 `0` geven. Dichte nulmask-GEMM is alleen
een kwaliteitsevaluator en geen runtimebenchmark.

## Gates

Een spread-cel is primair positief bij 25% wanneer:

- relatieve CE-toename `<2%`;
- gemiddelde finale KL `≤0,01`;
- top-1-overeenkomst `≥95%`;
- exacte 100%-control.

De spreadfase is alleen positief wanneer **alle 12 laag×domeincellen** slagen.
De moonshot is 10% met relatieve CE-toename `<3%` in alle 12 cellen en wordt
afzonderlijk gerapporteerd.

Harde falsificatie: een 25%-cel met relatieve CE-toename `≥2%`, KL `>0,02`,
top-1 `<90%` of een falende exact-control. Een tussenuitkomst is inconclusief.
Bootstrapintervallen zijn verplicht maar wegens twee blokken niet gatevormend.

Alleen een volledig positieve spreadfase opent een afzonderlijk
gepreregistreerde gelijktijdige full-depth-oracle. Geen predictor, index,
packed kernel of Eureka-claim vóór die stap.
