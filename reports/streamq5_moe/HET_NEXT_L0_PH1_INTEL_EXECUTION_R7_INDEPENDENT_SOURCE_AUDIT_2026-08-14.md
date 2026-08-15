# PH1 Intel execution R7 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only source audit. No preflight, payload, compiler, OpenCL, or device call was executed.

## Verdict

**NO-GO for executing the frozen R7 static preflight.**

The verifier correction itself is sound, the provenance chain is frozen, and the positive all-row sentinel is correctly specified. However, the write-after-loop negative control is not deterministic, and the two lifecycle-labelled gates exercise the old R6 runner rather than the new R7 namespace. A bounded preflight-only R7P revision can close both issues without changing R7 verifier arithmetic or backend/common code.

## Frozen identities and absence

Observed SHA-256 values match the handoff:

- runner: `8fae33f8fab9b3331caef23b04895990c0ea270e7f42b1cc7f23d2e4a8fe566a`
- verifier: `549e3d9574ebea0d5c58f2d919608dee2c0f211ea9acbafad678670cebf2a39d`
- preflight: `7b4e5552ef0ddd38ed0db40ca711bd1e1c9ece84c1345766709a66ceb2dd61c6`
- preregistration: `de3338dd017f552cd3c997e11d2eb4fb95e41fbdb64b9238d931b4702f21a2c7`
- closed lock: `9ea84555ee7e09b2fea68d2f5e3a3c3c0591a8d8b62ad652b7e946c4d0fcb36f`
- reused R6 backend: `8bbfa1a69caef5bb78f0a320f3f9093d2e778fa7f8ed67f8e67026ed0b87861f`
- reused R6 common: `d6abe5792e3069c15cef87f8b8550bb8d9893f992fd7bb93a71e0264d34890e1`

The R7 output directory, R7 static-preflight result, and physical result are absent. The lock is closed/PENDING and correctly binds the immutable R6P1 15/16 result `0b368d3e...`, its source audit `509f3d07...`, and its diagnosis `36377ee6...`.

## Checks that pass source audit

1. **Verifier repair and scope.** Direct R6-to-R7 diff confirms `out[row]=rb(lanes[0])` moved inside the row loop. The integer FMA, BF16 rounder, reduction trees, codec, production shapes, stage hashes, and `verify_dict()` gates are unchanged. Other verifier differences are R7 path/kind/provenance changes plus the new sentinel helper.

2. **Runner scope.** Direct R6-to-R7 diff contains mechanical output/failure/quarantine/lock/prereg/preflight/verifier names, R7 bundle kinds, R6P1 provenance bindings, and the new authorization token. The backend and common modules remain byte-identical.

3. **Positive all-row sentinel.** Both production shapes are exercised: 512 rows × width 2048 and 2048 rows × width 512. Every row has an exact selected BF16 word (`0x3f80`/`0x4000`) at column zero, the one-hot input is BF16 `0x3f80`, and every other word is zero. Two fresh full evaluations per shape must equal independently constructed expected bytes, with first/second/expected SHA-256 equality and exact row/byte evidence.

4. **Full verifier fixture.** The baseline uses full production tensor/output/counter shapes, independently decodes the wire records into nonzero q=1/scale=1 weights, requires `baseline_false_names=[]`, and only then executes exactly 28 named mutations through the actual independent R7 verifier. The rejected-name list must contain all 28 names.

5. **No hidden device path in the intended preflight.** The AST loader check remains call-node based and non-vacuous; the backend device APIs occur only behind runtime construction, not at module import.

## Blocker 1 — negative mutant depends on uninitialized memory

`write_after_loop_negative()` obtains the corrected `linear()` source, moves the assignment outside the row loop, executes that mutant, and calls `linear_all_row_sentinel()` on it. The mutant still allocates output with `np.empty(r,np.uint16)`. It therefore leaves rows `0..r-2` uninitialized—the exact defect under test—and the negative gate passes only if those arbitrary bytes differ from the expected array.

This is not deterministic. The positive sentinel runs immediately before the mutant and frees arrays with the same 512- and 2048-word sizes containing exactly the expected alternating words. NumPy/allocator reuse can return one of those blocks to the mutant. In that case the unwritten rows already equal the target; the mutant can spuriously pass the positive sentinel, making the intended negative-control gate fail. Repeated digests do not repair this because the second `np.empty` allocation can reuse the first mutant output.

Required R7P repair: poison the mutant output allocator deterministically before executing the bad function—for example, inject a local NumPy proxy whose `empty(r,uint16)` returns a full fixed poison word not present in the target—or rewrite only the mutant's allocation to a fixed poison fill. Then require exact poison retention in rows `0..r-2`, correct final-row assignment, positive sentinel failure, and repeated deterministic mutant digests. Also retain a structural AST gate proving the mutated assignment is outside `for row` while the frozen R7 assignment is inside it.

## Blocker 2 — R7 lifecycle labels test R6, not R7

The R7 preflight imports both runners, but its `transaction_failure` and `production_lifecycle` checks call the inherited R6 helpers with `run6`. Those helpers hard-code R6 result/manifest/commit/failure kinds. Thus the gates prove the already-audited R6 lifecycle again; they do not execute the R7 output paths, R7 kinds, R7 failure kinds, or R7 recovery behavior introduced by the namespace revision.

The manual source diff shows no hidden lifecycle/science change, so this is a bounded coverage defect rather than evidence of a production bug. Nevertheless, a gate labelled as R7 transaction/lifecycle coverage must exercise the actual R7 module.

Required R7P repair: add R7-specific TEMP-only transaction and lifecycle simulations using R7 kinds and the actual R7 runner, covering valid commit, stale/corrupt temp, authorization failure, resource failure, payload failure, post-device structured failure, telemetry evidence, oversized quarantine, valid completed output, cleanup/restoration, and no real payload/device call. Keep the inherited R6 gates only as historical regression evidence if desired, not as substitutes.

## Authorized next step

Freeze a preflight-only R7P revision binding this audit, with deterministic poisoned negative-control evidence and actual R7 TEMP lifecycle simulations. Runner, verifier arithmetic, backend, common, payload, thresholds, and device path need no scientific change. After a new independent source audit, exactly one no-device R7P preflight may be considered; physical execution remains closed.
