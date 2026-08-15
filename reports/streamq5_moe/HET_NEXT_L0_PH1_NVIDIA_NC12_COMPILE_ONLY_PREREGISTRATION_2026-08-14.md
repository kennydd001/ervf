# HET-NEXT-L0 PH1 NVIDIA NC12 compile-only preregistration

Status: immutable design-only; implementation and execution remain closed. NC12 supersedes NC11 only for the two clusters in independent audit SHA `b283bb6d1782226dc175499770a331426ed8797e48a5b9f45aae66f82b15d205`.

The cap-minus-one and cap manifest fixtures are loader-only. Each terminates as `loader_accepted_no_compile`, exit zero, publication `none`, `attempt_consumed=false`, `next_invocation_allowed=false`, `compiler_loaded=false`, and has ten ordered NVRTC ledger rows with `attempted=false`. They never become `compile_positive` and create no compile evidence.

Historical NC10 durability is fully specified independently: exact path, schema, 4 MiB cap, required keys, absence baseline and file, directory, orphan, over-cap and collision mutations. Historical NC10 static-preflight failure and quarantine roots each freeze their own exact schema/topology and five fixtures: baseline absence, missing declaration, extra file, corrupt record and collision. These augment rather than replace current NC11 symmetry.

All NC11 BOM, rejection, topology, environment, cache, artifact and transaction requirements remain unchanged. No run is authorized.
