# RSIV-MoE P1B — V2-long-prefix-preregistratie

Vastgelegd op `2026-08-11T07:54:09.1347211Z`, vóór de long-prefixcapture.

## Waarom dit geen reddingsvariant is

P1A blijft definitief `screen_negative_v2`: 96 prefix- en 32 futuretokens per
blok leveren geen bruikbare rank-32 fast path. RSIV is echter expliciet
gemotiveerd als **promptcompilatie bij T=1.024**, en alle atlasprojecties in het
bronrapport gebruiken die promptlengte. P1B verandert daarom geen P1A-resultaat,
rank, threshold of gate; het is een vooraf afgescheiden test van één reeds
gedocumenteerde variabele: langere prompt-specifieke prefixdekking.

## Bevroren opzet

- Model en revision: DeepSeek-V2-Lite Base,
  `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Dataset en revision: WikiText-2-raw-v1,
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Lagen: 1, 13 en 26.
- Natuurlijke top-6-routes; geen routewijziging of routerrenormalisatie.
- Twee onafhankelijke validationcontexten en twee onafhankelijke testcontexten.
- Contextlengte: 1.152 modeltokens.
- Per context bouwen posities `[0,1024)` afzonderlijke per-expert `Q/P`-bases.
- Alleen posities `[1024,1152)` zijn future-evaluatie.
- Contextbases worden nooit samengevoegd; een expert zonder prefixwaarneming
  telt op future als miss.
- De eerste 2.304 tokens uit de vooraf gepinde raw datasetstream vormen de twee
  contexten per split. Er wordt geen contextstart geselecteerd.

Validation en test mogen in één raw capture worden opgeslagen. De testoffsets
worden pas gelezen nadat één globale validationkandidaat plus capturehash in een
append-only selection-lock staat.

## Ongewijzigde grid en gates

```text
ranks = [4, 8, 16, 32, 64, 128]
thresholds = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
primary_rank_cap <= 32
double_gate_fast_fraction >= 0.92
projected_routed_cold_byte_reduction >= 10.0x
```

Eén rank en één gedeelde `rho_x/rho_z`-threshold gelden voor alle drie lagen en
beide validationcontexten. Bij meerdere passes kiest validation eerst de
kleinste threshold en daarna de kleinste rank. Zonder pass wordt één
diagnostische kandidaat gekozen op maximale
`min(double_fast/0.92, cold_reduction/10)`, met lagere threshold en rank als
tie-break. Test bevestigt uitsluitend die lock; de volledige testgrid wordt pas
daarna diagnostisch gepubliceerd.

## Byteboekhouding

Zoals P1A telt een inputmiss `2/3` volledige expertbytes en een `z`-miss `1/3`.
De geprojecteerde reductie is het omgekeerde van de gemiddelde koude fractie.
Atlasreads, basisprojecties, aandacht, shared experts, latency en kwaliteit zijn
niet inbegrepen. Dit is geen runtimeclaim.

## Verplichte controles

1. Route-ID's sluiten bit-exact met de officiële router; routergewichtfout
   maximaal `1e-6`.
2. Per laag/split zijn er exact `tokens × 6` invocations.
3. Per context geldt `rank(X_e), rank(Z_e) <= n_e` en de expert-count-
   cancellationbound.
4. De volledige opgeslagen-rankbasis reconstrueert alle prefix-`x/z` binnen
   FP64 relatieve L2 `1e-10`.
5. De reeds onafhankelijke P1A-operatorimage-identiteit voor `x/g/u/z/y` blijft
   een vereiste upstreamcontrol; P1B verandert geen operatoralgebra.
6. Het officiële full-weightfallbackpad blijft ongewijzigd.

Een controlfout maakt de poging ongeldig. Zij mag geen thresholdwijziging
veroorzaken.

## Besluitregels

- `long_prefix_screen_positive`: validation en test halen gezamenlijk de twee
  primaire gates voor dezelfde kandidaat. Alleen dan mag V2-P2 opnieuw als
  afzonderlijke preregistratie worden overwogen.
- `long_prefix_screen_negative_v2`: validation of test faalt met geldige
  controls. Dan is promptlengte geen verklaring voor P1A en gaat de volgende
  P1 uitsluitend naar de reeds aangekondigde hogere-E-familie.
- `invalid`: minstens één verplichte control faalt.

