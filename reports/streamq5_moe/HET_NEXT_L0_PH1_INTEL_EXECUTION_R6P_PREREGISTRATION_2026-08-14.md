# PH1 Intel execution R6P — preflight-fixture shape repair

Status: closed/PENDING. This revision authorizes no preflight, payload read,
compiler load, OpenCL load, or device call.

R6P is a preflight-only erratum. It binds the frozen R6 implementation, the R6
independent source audit, and the immutable R6 preflight-crash diagnosis. It
changes no production runner, backend, common code, verifier, kernel, codec,
buffer contract, launch geometry, thresholds, identity policy, or claim.

The sole repair is to make both synthetic oracle callsites conform to the
production verifier's fixed reduction widths:

1. `codec_oracle_fixtures` uses explicit `(1,512)` and `(1,2048)` BF16-word
   sentinels. The first eight words are `0x3f80`; all other words are zero; both
   exact expected sums are `0x4100`.
2. `verifier_mutations` uses exact production shapes: gate/up `[512,2048]`,
   down `[2048,512]`, input 2048 BF16 words, gate/up/silu/activation 512 BF16
   words, down 2048 BF16 words, and counters of 512/512/512/2048 little-endian
   `uint32(1)` values.
3. The fixture uses the frozen production `BUFF`, `ARGS`, `LAUNCH`, three
   675,840-byte records, and an explicit schema assertion before invoking the
   independent verifier. No verifier linear function is replaced or patched.
4. All R6 ownership, lifecycle, resource, transaction, bundle, provenance,
   control, and mutation checks remain unchanged by importing their frozen R6
   implementations.

The prior R6 crash is classified as a valid no-device infrastructure negative,
not a scientific or Intel result. R6P must be independently source-audited
before its single closed static execution may be authorized.
