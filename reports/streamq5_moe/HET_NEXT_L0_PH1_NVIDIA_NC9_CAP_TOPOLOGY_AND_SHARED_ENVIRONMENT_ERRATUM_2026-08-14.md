# NC9 cap, topology and shared-environment erratum

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc9_fixture_manifest.json` is normative.

Its `caps.fixture_manifest` is `8388608`; the frozen byte size must satisfy `0 < bytes <= 8388608`. No output cap changes. `inherited_topology.nc8_durability_adjudication` freezes the exact path, requires absence, has an empty allowed-terminal set and names file, directory and temp-presence mutations.

`shared_contract.module` and the design lock reserve the same NC9 path. The three environment functions and five classifier/transaction functions are exact exports. Static preflight and runner must compare module path/hash, object identity and code-object identity before calling them. The manifest includes a positive identity fixture and negative copy, monkeypatch, wrong-order and early-publication fixtures. The shared functions themselves execute all fifty literal NC8 environment rows; a snapshot or duplicate helper is not acceptable.

NC8 remains otherwise byte-for-byte authoritative through its direct bindings and audit chain.
