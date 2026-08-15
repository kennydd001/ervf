# NC16 typed-value and source-lock design

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc16_typed_schema_freeze_fixture_manifest.json` is normative. It contains 137 current roots and 1,093 complete cases across design, implementation-freeze and runtime.

Validation order remains descriptor, stage, node, cap, identity, typed schema and terminal composition. A source identity cannot be inferred from the observed file or a synthetic table: it must resolve exactly once from `synthetic_nc16_source_lock`, whose literal structure represents the future immutable source-lock input contract. Static preflight executes every typed-key mutation and all lifecycle compositions using the production shared functions.
