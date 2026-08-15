# HET-NEXT-L0 PH1 NVIDIA NC18 legacy-authority cleanup preregistration

Status: immutable design-only. Implementation, preflight, compiler, payload, and device execution remain closed. NC18 binds the NC17 independent design audit SHA-256 `bb900dd1153e489e71d2576aa8d260315632bd222d9d4acc7970e0bb547a1906` and changes only the two defects named there.

The normative manifest has no top-level `synthetic_source_bindings`, `synthetic_nc16_source_lock`, or `implementation_freeze_source_lock_contract`. Those spellings are permitted only as forbidden-key declarations and exact rejection-case mutation operands. No NC16 implementation-freeze authority remains normative.

The sole source identity authority is the observed NC18 source-lock document. Its bootstrap-bound raw bytes are parsed directly. It must have revision `NC18` and exactly 32 unique `source_identity_entries`; no side-channel mapping is accepted. Rejection cases cover each legacy key/root, revision `NC16`, and count `28`.

The observed entry field declaration is exactly this ordered unique eight-name vector: `path`, `node_type`, `size`, `sha256_or_null`, `children`, `schema_key_values`, `parse_status`, `content_base64_or_null`. Both count and unique count must equal 8. Duplicate, dropped, extra, and reordered declarations are rejected.

All NC17 stage, authority, topology, identity, and mutation contracts otherwise remain unchanged. No run is authorized.
