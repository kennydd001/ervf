# PH1 Intel execution R7P — deterministic regression and R7 lifecycle preflight

Status: closed/PENDING. No preflight, payload, compiler, OpenCL load, or device
call is authorized.

R7P is preflight-only and binds immutable R7 plus independent audit SHA-256
`20afdd0e535c31c69ce9ae295ec2cded94ffcd1d15a4c156f258de175c6d32bb`.
It changes no runner, backend, common, verifier, kernel, codec or claim.

Two preflight repairs are frozen:

1. The write-after-loop regression mutant replaces `np.empty` with deterministic
   BF16-word poison `0x7e00`. For both 512×2048 and 2048×512 sentinels, rows
   `0..r-2` must remain poison, the final row must equal its correct expected
   word, two bad runs must have identical full-byte SHA-256 digests, and that
   digest must differ from the expected target. AST inspection must prove the
   R7 assignment is inside the row loop and the mutant assignment is outside.
2. TEMP-only transaction and production-lifecycle simulations invoke the actual
   R7 runner and R7 result/manifest/commit/failure kinds. They cover existing
   valid commit, stale recovery/quarantine, atomic failure, auth failure with no
   filesystem write, RAM failure, payload failure, ordinary post-device failure,
   primary-plus-telemetry failure, oversize quarantine/cap, and completed
   negative return without failure pollution.

All R7 all-row positive sentinels, full verifier baseline map, exact 28 named
mutation rejections, control, ownership, resource and no-device checks remain.
