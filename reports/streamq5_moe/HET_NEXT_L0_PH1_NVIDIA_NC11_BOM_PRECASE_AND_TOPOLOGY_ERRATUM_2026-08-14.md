# NC11 BOM, pre-case and topology erratum

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc11_fixture_manifest.json` is normative and itself is serialized with exactly one leading UTF-8 BOM. `manifest_loader_contract.canonical_file_encoding` specifies the raw/BOM/strict-decode order. The four size fixtures and four BOM/schema fixtures retain exact stat, read, parse, compiler-loaded and disposition observations.

Pre-case rejection happens before DLL load or any NVRTC call. Its exact ten-row ledger uses the normal operation order and `attempted=false,error=not_attempted:manifest_precheck` throughout. Accepted size fixtures remain loader-only evidence and likewise do not authorize compilation.

`inherited_topology` names historical NC8, NC9 and NC10 roots and the current NC11 durability root. The lock independently enforces their absence and closes all four NC11 verifier terminal namespaces.
