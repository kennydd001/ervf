# HET-NEXT-L0 PH1 NVIDIA NC15 semantic-identity preregistration

Status: immutable design-only; implementation and execution remain closed. NC15 supersedes only the two identity gaps in NC14 independent audit SHA `53f9d310cb641e21424bdec437edde4186f341bd8c9aaf940a3d15bfcecf7d64`.

Every observed root record now has exactly `path,node_type,size,sha256_or_null,children,schema_key_values,parse_status`. A root JSON file is schema-valid only when parsing succeeds and the key set, kind, status and value types exactly match its descriptor. NC15 contains runtime-positive fixtures plus missing, extra, wrong-type, wrong-value and invalid-JSON fixtures for all 24 current lock/preflight-result file roots.

Every immutable source root has `identity_policy=bound_size_sha256`, an exact `binding_key`, and the stage policy: design requires absence; implementation freeze must resolve expected bytes and SHA-256 from the exact source lock before topology; runtime requires both resolved values. Descriptor placeholders remain null during design and cannot authorize presence. A synthetic bound table drives positive cases for all 24 current source roots and exact wrong/missing/extra-binding, byte-drift and hash-drift negatives.

Other roots explicitly select `schema_only`, `content_hash` or `none`. Classification order is descriptor, stage, node identity/type, multiplicity, cap, identity, schema, then terminal composition. Thus a syntactically plausible design-phase file is a stage-policy violation, never silently trusted.

All NC14 path, pattern, semantic-role and historical equality requirements remain unchanged. No run is authorized.
