# E100-MRHS V3 addendum — adopted single-RHS baseline

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: **no E100-MRHS/MRHS256 target-GPU result exists**.

This addendum corrects the performance reference before first target execution. It does not change any numerical pass threshold in the earlier MRHS preregistrations.

## Audit finding

The first MRHS runners timed BF16/F32/FP8 candidates against `GPUKernels.mv_*`, the original 256-thread production GEMVs. That is a valid exact correctness reference but no longer the actual V6 performance baseline.

The adopted V6 selective policy is already frozen in `selective_ervf_v3.py`:

- BF16 ERVF only for `(4096,2688)` and `(2688,4096)`;
- FP8 ERVF only for `(10304,2688)` and `(2688,4096)`;
- all other BF16/FP8 shapes remain production;
- F32 remains production-only;
- NVFP4 already uses the adopted FusedNVFP4 ERVF path.

Therefore comparing MRHS against the slower legacy GEMV on selected shapes could overstate integration value.

## V3 rule

Every MRHS32/MRHS256 correctness batch must now establish:

1. original production single-RHS output;
2. adopted selective single-RHS output;
3. candidate multi-RHS output;
4. deterministic candidate repeat.

Mandatory bitwise equalities are `production == adopted == candidate == candidate_repeat` for every RHS/output element. This is stronger than either earlier runner alone.

Performance `REF` is **only N sequential calls through the adopted selective single-RHS dispatcher**. It is no longer the legacy production kernel on shapes where V6 uses DenseERVF.

The A/B timing order, drift limits, thresholds and claim boundaries remain unchanged.

## Case-set correction before data

MRHS32 V3 adds Mamba output projection in its actual stored kind. Mamba output is a common-weight matrix and is part of the adopted full runtime; omitting it would make the weighted common-matrix summary incomplete. This addition is made before any E100 target result and does not alter the already-frozen N=4 thresholds.

The explicit family support gate remains the same seven families from the first preregistration (`attention_q`, `attention_o`, `router`, `mamba_in`, `shared_up`, `shared_down`, `lm_head`). Mamba-out is an additional measured family and cannot compensate for a missing frozen family.

MRHS256 retains its stricter eight-family requirement including Mamba-out.

## Interpretation

Only V3 adopted-baseline result files are eligible for E100 integration decisions. V1/V2 source files remain as an audit trail but their performance numbers, if accidentally run later, must not be used to claim an E100-worthy primitive.
