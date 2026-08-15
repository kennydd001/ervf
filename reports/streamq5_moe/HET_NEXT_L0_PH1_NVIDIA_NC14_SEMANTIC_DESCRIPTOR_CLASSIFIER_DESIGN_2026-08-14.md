# NC14 semantic descriptor classifier design

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc14_semantic_topology_fixture_manifest.json` is normative. It contains seven complete descriptors, six historical lock expectations, the exact 99-path NC14 set and 242 executable cases.

The classifier validates descriptor/root/pattern field sets first, then path normalization and uniqueness, then observed node identity/type/multiplicity, caps and child schema, and only then terminal composition. Terminal-positive, terminal-negative, failure-history, quarantine, durability, result and lock roles retain distinct semantics. Only an exact declared terminal or one recoverable in-progress entry can be valid outside fresh state.

Static preflight must call the production shared functions on every literal case and independently compare every historical pathset digest. Snapshot predicates or revision-name branches are forbidden.
