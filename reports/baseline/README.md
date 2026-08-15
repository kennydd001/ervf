# Baseline-reportindex

Deze map bevat zowel definitieve als tussentijdse JSON-rapporten. Gebruik voor
het Eureka-verdict uitsluitend de hieronder gemarkeerde bestanden; eerdere
bestanden blijven staan voor auditability en worden niet stil overschreven.

## Authoritatieve eindrapporten

- `paper_context1024_wikitext_cache_prior_aggressive.json` — paper-faithful
  reproductie van de sterke Cache-Prior-baseline bij context 1.024.
- `layer26_route_equivalence_full.json` — volledige 1.024+1.024-token,
  924-routes-per-token equivalentiemeting.
- `layer26_conformal_cache_selector_full.json` — held-out predictor- en
  conformal-safetyfalsificatie.
- `preregistered_wikitext_offset4096_mass_budget_confirmation.json` — vooraf
  vastgelegde 2.048+2.048-tokenconfirmatie; alle vijf gates geslaagd.
- `preregistered_mass_budget_confirmation_gates.json` — machineberekende
  verificatie van de vijf vooraf vastgelegde gates.
- `mass_budget_cache_accounting.json` — geselecteerde vaste-prior/Mass-Budget-
  vergelijkingen en packed-int4-byteprojecties.
- `matched_cache_policy_kv_rollout_4tokens.json` en
  `matched_cache_policy_kv_rollout_delta12_14_4tokens.json` — autoregressieve
  tegenproef; tokenstabiliteit slaagt, universele dominantie niet.

- `layer26_behavioral_observability_reliability.json` — betrouwbare
  acht-sample observability-falsificatie.
- `layer26_dynamic_precision_exact_oracle.json` — exact 3→4-bit-oracle.
- `layer26_dynamic_2to4_exact_oracle.json` — exact 2→4-bit-oracle.
- `layer26_dynamic_precision_predictors.json`,
  `layer26_quadratic_mask_predictor.json` en
  `layer26_progressive_bitplane_predictor.json` — teacher-free
  predictorfalsificaties.
- `layer26_route_equivalence.json` — uitputtende top-12-choose-6-test.
- `layer23_route_equivalence_downstream.json` — downstreaminterventie.
- `modelwide_cache_aware_bottom1_teacher_lru_capacity32_wikitext_1024_ci.json`
  — definitieve WikiText-run met teacherroutes als strict baseline.
- `modelwide_cache_aware_bottom1_teacher_lru_capacity32_diverse_1024_ci.json`
  — definitieve instructie/code-run met teacherroutes als strict baseline.
- `cache_aware_teacher_lru_kv_rollout_4tokens.json` — definitieve korte
  KV-cacherollout met teacherroutes als strict baseline.
- `route_cache_storage_io_accounting.json` — storage- en I/O-projectie op basis
  van de twee definitieve modelbrede runs.

De nieuwste narratieve synthese staat één map hoger in
`MASS_BUDGET_EUREKA_2026-08-10.md`. Het eerdere brede behavioral-compressie-
verdict blijft in `EUREKA_VERDICT_2026-08-10.md` staan.

## Authoritatieve exploratieve frontmetingen

- `confirm8_wikitext_cache_routing_pareto.json` — sterke Max-Rank, Cumsum en
  vaste Cache-Prior-baselines op 8×128 tokens per split.
- `matched8_wikitext_mass_budget_vs_cache_prior.json` — fijne vaste-λ versus
  δ-vergelijking op het eerste WikiTextvenster.
- `matched8_diverse_mass_budget_vs_cache_prior.json` — instructie/code-
  domeintransfer.
- `matched_context1024_wikitext_mass_budget_vs_cache_prior.json` — één lang
  1.024-tokenblok per split; nuttig maar zonder betrouwbaar bootstrapinterval.

## Bewust bewaarde voorlopige rapporten

De bestanden met `modelwide_cache_aware_bottom1` maar zonder `teacher_lru` en
de rollout `cache_aware_exact_lru_kv_rollout_4tokens.json` gebruiken voor hun
strict loadcount de ongewijzigde route op de licht afwijkende studentstate.
Hun kwaliteitsmetingen blijven bruikbaar, maar hun loadpercentages zijn
vervangen door de authoritatieve teachercache-runs hierboven.

Bestanden zonder `exact_lru` stammen bovendien van vóór de correctie waarbij de
volledige within-token LRU-volgorde werd doorgerekend. Zij mogen niet worden
geciteerd als eindresultaat.
