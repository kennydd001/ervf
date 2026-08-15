# NC12 loader-only and NC10-symmetry erratum

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc12_fixture_manifest.json` is normative. Its accepted boundary observations remain BOM-inclusive and hash-identical to NC11, but their terminal and ledger are now purely loader-only.

`inherited_topology.nc10_durability_adjudication`, `.nc10_static_preflight_failures` and `.nc10_static_preflight_quarantine` contain complete literal schemas. The manifest contains 6 + 5 + 5 corresponding cases, and the design lock preserves the full NC11 inherited/current absence union.
