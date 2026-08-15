# NC10 topology and manifest-boundary erratum

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc10_fixture_manifest.json` is normative. `manifest_loader_contract` freezes the loader, exact 8 MiB cap, four boundary streams, derivation, hashes, byte-read counters and parse counts. Cap-minus-one and cap streams contain the complete ASCII JSON `{"kind":"fixture_size_sentinel"}` followed by spaces to their exact sizes; trailing spaces are JSON whitespace. Over-cap and empty fixtures are rejected before open.

`inherited_topology` retains historical NC8 durability absence and adds current NC9 preflight-failure, preflight-quarantine and durability roots. The current durability schema is named and capped, but its presence is invalid in the NC10 fresh state. The manifest contains positive absence and independent file, directory, orphan and oversize mutations.

NC9 remains otherwise directly bound and authoritative.
