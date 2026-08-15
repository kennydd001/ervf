# H2 Speculative Route Coalescing — preregistratie laag 26

Vastgelegd op `2026-08-10T11:30:36.8509214Z` vóór implementatie en vóór
inspectie van block-unionuitkomsten. De onderliggende route-KL-tabel bestond al
en is eerder geaggregeerd gerapporteerd; blockoptimalisaties, slates en
blockmetrics zijn nog niet berekend.

## Hypothese

Als routes voor een toekomstig verificatieblok tegelijk bekend zijn, kan één
functioneel equivalente top-6-route per token gezamenlijk worden gekozen zodat
de unieke routed expertset minstens 40% kleiner is dan de natuurlijke route-unie
en minstens 25% kleiner dan een sterke sequentiële per-token Mass-Budget-keuze,
bij gemiddelde lokale teacher→candidate-KL maximaal `0,001`.

## Bevroren input

- `data/traces/layer26_route_equivalence.safetensors`;
- DeepSeek-V2-Lite revision
  `604d5664dddd88a0433dbae533b7fe9472482de0`;
- WikiText-2-raw-v1 revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- eerste 256 validatie- en eerste 256 testtokens, ieder twee onafhankelijke
  sequence blocks van 128 tokens;
- per token de 924 combinaties `top12 choose 6`, met oorspronkelijke
  ongenormaliseerde routergewichten en exacte volledige-vocabulaire lokale KL;
- de natuurlijke eerste zes posities worden altijd in een slate behouden,
  ook als numerieke reconstructie-KL net boven een drempel ligt.

Validatie bepaalt uitsluitend of de vooraf gekozen kandidaat door mag. Test
verandert geen drempel, cap, bloklengte, tie-break of algoritme.

## Slates en blokken

Volledige vaste matrix:

- KL-drempels `{1e-4, 1e-3, 3e-3}`;
- slatecaps `{16, 32, 64}`;
- niet-overlappende verificatieblokken `{2, 4, 8, 16}` die nooit een
  128-token-sequencegrens kruisen.

Eligible routes worden stabiel gerangschikt op `(KL, subset_index)`. De
natuurlijke route vervangt zo nodig de laatste route binnen de cap. De primaire
cel is **blok 8, drempel `1e-3`, cap 32**.

## Optimalisatie en baselines

De exacte ILP heeft binaire `x_(t,r)` en `y_e`, kiest exact één route per token,
dwingt `x_(t,r) ≤ y_e` voor elk route-expert af en minimaliseert `Σ y_e`.
SciPy 1.18.0/HiGHS moet status optimal geven en de geëxtraheerde route-unie moet
exact gelijk zijn aan de objective. Exacte ILP wordt uitgevoerd voor alle
4-token sweepcellen en voor iedere primaire 8-tokenblock.

Vaste approximaties/baselines:

1. natuurlijke top-6-routes;
2. per-token Mass-Budget `δ=0,004`: binnen dezelfde slate en mass-lossgrens
   sequentieel minimaal nieuwe experts, daarna mass loss, KL en subsetindex;
3. fixed Cache-Prior: de reeds op 2.048 WikiText-validatietokens gekalibreerde
   32 hot experts van laag 26; per token minimaal cold experts, daarna KL;
4. marginal-union greedy: per token minimaal nieuwe block-unionexperts, daarna
   KL en subsetindex;
5. eligible-set pruning: experts op slate-aanwezigheid, kleinste stabiele prefix
   die voor ieder token minstens één volledige route bevat, daarna laagste KL;
6. beam-DP met vaste breedte 1.024, deduplicatie op 64-bit unionmask en
   rangschikking `(union_count, total_KL, union_mask, route_path)`;
7. exacte ILP-oracle.

Beam moet op de exact opgeloste cellen zijn optimale unioncount kunnen worden
vergeleken; een mismatch blijft zichtbaar en wordt niet stilzwijgend met een
grotere beam vervangen.

Een secundaire fixed-cache-analyse minimaliseert
`union(routes) \ hot_cache32`; zij verandert de lege-cache-primary niet.

## Metrics en accounting

Per block en methode worden vastgelegd: gekozen subsetindices en expertsets,
unieke unioncount, cold unioncount, gemiddelde/som lokale KL, gewijzigde
tokenfractie, natuurlijke-route-Jaccard en router-mass loss. Aggregaten gebruiken
gepaarde 10.000× block-bootstrap, seed `20260810`, voor:

- `1 - Σ(method_union) / Σ(natural_union)`;
- `1 - Σ(oracle_union) / Σ(mass_budget_union)`;
- gemiddelde lokale KL.

BF16/int4-expertbytes zijn exacte deterministische accounting uit
`3×1408×2048` gewichten per expert. Dit is geen cache-runtime, speculative
acceptatiemeting, transfermeting of snelheidsclaim.

## Gates en stop/go

De layer-26-primary slaagt alleen als op **validatie én test**:

1. exacte-ILP unionreductie versus natural `≥40%`;
2. exacte-ILP additionele unionreductie versus Mass-Budget `≥25%`;
3. geselecteerde gemiddelde lokale KL `≤0,001`;
4. originele routecontrol aanwezig is, HiGHS iedere primary block optimaal
   oplost en union/objective exact sluiten.

Harde falsificatie: minder dan 25% exact-oracle-unionreductie versus natural op
één split, minder dan 10% additionele reductie versus Mass-Budget op beide
splits, gemiddelde KL boven `0,001`, of een falende exacte control. Een
tussenuitkomst is inconclusief negatief en opent evenmin downstreamwerk.

Alleen een volledig positieve layer-26-primary opent een afzonderlijk
gepreregistreerde laag-23-interventie plus exacte lagen 24–26 met CE `≤0,5%` en
top-1 `≥99%`. Q3/Q4-bitplanes en H3-atomtiles worden alleen na die lokale gate
bekeken. De latere `≥2×` echte speculative expertbytes per geaccepteerde token
blijft geblokkeerd zonder fysieke verificatieruntime. Er wordt geen
noveltyclaim gedaan.
