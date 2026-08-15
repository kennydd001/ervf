# LDHERA-MoE P0A — training-geleerde laaglokale cache

**Vastgelegd:** 2026-08-11, vóór berekening van slotallocaties of uitkomsten.

## Hypothese en datastatus

De vaste 48-primary + 8-victimcache verdeelt capaciteit vrijwel uniform, maar
cold-churn kan per laag verschillen. LDHERA verdeelt dezelfde 56 BF16-slots
domeinspecifiek over de 48 lagen. Alleen HERA-trainingroutes bepalen de
verdeling. De DHERA-validationroutes zijn al geopend; P0A is dus exploratief en
een positief resultaat vereist een nieuwe blinde P0B.

## Enige allocatieregel

Voor elk bekend domein:

- gebruik de reeds gelockte DCHERA-basis van 4.280 experts;
- bereken voor iedere laag op de 32 trainingscontexten de exacte LRU-misses
  voor capaciteiten 0–56, met reset per 1.024 tokens en top-k-volgorde 0–7;
- kies via exacte dynamische programmering gehele capaciteiten `c[0..47]` met
  som exact 56 die het totale aantal training-cold-misses minimaliseren;
- bij gelijke totale misses wint lexicografisch de allocatie met meer slots in
  de laagste laagindex;
- lock de vijf allocaties vóór de validation-simulatie.

Tijdens validatie heeft iedere laag een eigen volledig associatieve LRU met de
gelockte capaciteit. Er is geen globale victimcache, prefetch, lookahead,
routewijziging of allocatiesweep.

## Verkeer en gates

Iedere context draagt conservatief één volledige geprojecteerde entropy-base-
transfer. Iedere cold miss kost 9 MiB. Memory-, mean-, p95- en p99-gates zijn
identiek aan DCHERA en moeten in elk domein slagen. Discrete nearest-rank wordt
gebruikt; onafhankelijke reproductie is verplicht.

P0A bewijst geen echte packgrootte, transfer/repacktijd, overlap, kwaliteit,
domeinclassifier of tokens/s.
