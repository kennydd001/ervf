# PH1 Intel execution R6P1 — `oracle_outputs` negative diagnosis

Date: 2026-08-14  
Scope: read-only source/result diagnosis; no payload, compiler, OpenCL, or device call was made.

## Verdict

The immutable R6P1 preflight result is correctly **negative (15/16)**, but its sole failure is a verifier implementation defect, not evidence against the Intel mechanism, Q5 codec, or PH1 arithmetic.

Result identity:

- `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r6p1_static_preflight.json`
- SHA-256 `0b368d3ed9c823405a821f96b59f58cbb8cb4b2c48fa1d19431ca3f88db0742f`
- only false top-level gate: `actual_verifier_mutations_full_shapes`
- only false baseline verifier check: `oracle_outputs`

## Exact root cause

In frozen verifier `scripts/streamq5_moe/verify_het_next_l0_ph1_intel_execution_r6.py`, SHA-256 `b6e4909fcaf4a9113b3682bb2a2c6efbe1ca744f9de1bf480412dbac9f81d041`, `linear()` allocates `out=np.empty(r,np.uint16)` and iterates `for row in range(r)` at source line 78. The assignment at line 93,

```python
out[row] = rb(lanes[0])
```

is indented at the same level as `for row`, not inside it. Consequently:

- all row arithmetic is evaluated;
- only the final row is stored;
- rows `0..r-2` retain indeterminate `np.empty` contents;
- this affects both production linear widths: `(r,c)=(512,2048)` and `(2048,512)`;
- fixture construction and verifier replay allocate separate arrays, so their uninitialized bytes need not match;
- `oracle_outputs` therefore fails even though the defined last row and all metadata/lifecycle checks agree.

The result is consistent with this cause: all other 15 preflight gates are true, and within the baseline all verifier checks except `oracle_outputs` are true. Mutation execution is intentionally skipped after the baseline fails, so this artifact does not adjudicate the 28 mutation rejections.

## Minimal immutable R7 revision

Do not edit or reinterpret R6/R6P/R6P1 in place. Create a fresh R7 verifier/runner/preflight/preregistration/lock namespace with:

1. the sole arithmetic source repair: indent `out[row]=rb(lanes[0])` inside the `for row in range(r)` body;
2. unchanged integer FMA, BF16 rounding, reduction trees, codec, buffers, launch contract, controls, thresholds, and physical runner science;
3. an AST/source-diff gate proving that this indentation/location change plus R7 path/hash bindings are the only verifier/science-adjacent changes;
4. two independent all-row sentinels using exact production shapes `(512,2048)` and `(2048,512)`, with a deterministic finite BF16 value assigned per row at one selected column, a one-hot BF16 input, and an independently constructed exact expected word for **every** row;
5. for each shape, at least two fresh `linear()` calls whose full byte arrays equal the independent expectation and whose SHA-256 digests are identical; record both digests and row counts (`512/512`, `2048/2048`);
6. a negative mutation of the repaired source/fixture that restores the write-after-loop behavior and must fail the all-row/determinism gate;
7. the existing production-sized baseline requirement (`baseline_false_names=[]`) followed by all exact 28 named verifier mutations, each independently rejected;
8. an immutable chain binding the R6P1 result SHA above, this diagnosis, all prior R6/R6P/R6P1 locks/audits/results, and the unchanged backend/common/compile/CPU evidence.

The all-row sentinel must test exact words, not merely `isfinite`, nonzero counts, or one final-row sample. Repeated full-array digests are necessary because the observed defect is uninitialized-memory nondeterminism.

## Downstream binding implications

- The frozen R6 verifier hash is embedded in the R6, R6P, and R6P1 locks. Replacing that file in place would invalidate the complete provenance chain.
- The R6 runner also binds the verifier path/hash. A physical R7 attempt therefore needs a path/hash-only runner revision and a fresh closed lock, even though backend/common/scientific arithmetic remain unchanged.
- The R6P1 negative result stays immutable and remains evidence that the old verifier/preflight failed. It cannot be relabeled PASS after the fix.
- No physical output exists, and no device was called, so there is no device result to discard or reinterpret.
- R7 may first authorize only a new no-device static preflight. Physical execution remains closed until that exact preflight passes and receives a separate authorization audit.

## Claim boundary

This diagnosis proves a verifier storage/indentation defect explains the R6P1 `oracle_outputs` failure. It does not prove PH1 Intel correctness, performance, or model-level quality. Those claims still require a clean, newly authorized physical R7 execution and independent artifact verification.
