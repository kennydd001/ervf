# HET-NEXT-L0-PH0-R3 — implementation design

Date: 2026-08-13  
Status: **design only; no runner, preflight, import, compiler or device call authorized**

This incorporates the complete PH0-R2 implementation design, SHA `f5e2b6336d16843912c96f9239bb077d406795870bbf595ac131a687c49cf5b7`, with one repair only.

The future standalone safe checker must implement this exact non-configurable control flow:

```text
size
structural_header_without_requested_expert_comparison
crc_codes_scales
exhaustive_field_range
official_source_and_pristine_codes_scales_digests
requested_layer_expert_projection_shape
frozen_input_identity_and_digest
canonical_full_record_digest
dispatch
```

No function argument, fixture, environment variable or control branch may replace canonical digests. The wrong-identity unit test constructs the exact expert-51 header and asserts its frozen header/record SHAs, pristine payload digests, stage trace ending at `requested_identity`, absence of later stages, and zero device submissions.

The independent verifier reconstructs the wrong-identity record from the pristine record, recomputes both new SHAs and the full stage trace, and rejects any runner-supplied aggregate verdict. The static preflight uses AST/control-flow inspection and mutation tests to prove the stage order, no override path and dispatch unreachability.

Every other R2 design obligation remains unchanged. Passing audit or preflight is not device authorization.
