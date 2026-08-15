# HET-NEXT-L0 PH1 NVIDIA NC13 generic-topology preregistration

Status: immutable design-only; implementation and every execution phase remain closed. NC13 supersedes the per-revision topology enumeration only, addressing NC12 independent audit SHA `9534241cb6e0eda6682b7bfc1642923ddff93dc663ad366dd393d9630ea97149`.

The future stdlib-only, import-inert shared module is exactly `scripts/streamq5_moe/het_next_l0_ph1_nvidia_nc13_compile_contract.py`. It exports `paths_for_revision(descriptor)` and `classify_topology(descriptor, observed_entries)`. Runner, preflight and verifier use the same hash-bound function and code objects. Source code may not branch on NC8, NC9, NC10, NC11, NC12 or NC13; revision differences exist only in immutable descriptors.

Each descriptor has exactly `revision,prefix,phase_roots,inprogress_patterns,caps`. `prefix` is the ordered set of namespace tokens whose four scripts and three locks are generated from fixed templates. `phase_roots` contains exact arrays for preflight result/failure/quarantine/durability, compile positive/negative/failure/quarantine and verifier positive/negative/failure/quarantine. Paths are forward-slash normalized, reject dot/traversal/duplicates and return ordinal-unique paths plus patterns.

The NC13 descriptor generates exactly the 80-path absence set bound by the NC13 lock. NC8-NC12 descriptors must reproduce the exact expected-absence set and canonical digest of their frozen locks. Drop, duplicate, wrong-revision and wrong-path mutations fail for each.

The generic classifier matrix covers fresh state; file and directory shape at every one of 80 generated paths; valid terminal; one recoverable debris entry; missing descriptor data; orphan, collision, mixed, multiple and over-cap states. Only the declared fresh, exact valid-terminal and single-debris outcomes are valid.

All NC12 compile-only, BOM, loader, environment, cache, artifact and lifecycle contracts otherwise remain directly bound and unchanged. No run is authorized.
