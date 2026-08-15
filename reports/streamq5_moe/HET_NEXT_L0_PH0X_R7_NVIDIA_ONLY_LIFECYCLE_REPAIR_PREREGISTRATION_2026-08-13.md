# HET-NEXT L0 PH0X-R7 — NVIDIA-only lifecycle repair preregistration

Date: 2026-08-13. Exploratory completion arm only; no formal PH0 pass.

R7 binds PH0X-R6 runner SHA `a1369c314a4e1367fa4ce3584555a7dc4db30ed9480cbdff289aa18af8417bdf` and prereg SHA `7e5c0ad01797120c66ce140f32207ed3460821aa3a0f4acbd6aff8f5a8231732`. It retains every frozen R5/R6 scientific input, Q5 record, CPU oracle, CUDA source, launch, buffer, comparison, identity and claim boundary.

Only three lifecycle/evidence repairs are allowed:

1. Track successful primary synchronization. A clean run makes exactly one synchronization call/ledger row; the `finally` synchronization is attempted only when primary sync was not completed, and its attempt/result is retained only in partial failure evidence.
2. Add each pinned allocation to the ownership list immediately after host allocation and before device allocation, so a device-allocation failure cannot orphan it. A complete allocation ledger row remains emitted only after its paired device allocation succeeds.
3. Cleanup errors cause structured failure carrying the full partial NVIDIA evidence. A completed NVIDIA evidence object is attached to the outer result before exact-ledger validation, so even validator failure preserves it.

Success retains the exact 24-row R6 ordered ledger. One new clean output directory, one NVIDIA-only attempt, no Intel API, retry or retuning. Claim remains one real projection/input only; no full expert/layer/model/performance/concurrency/deployment/novelty/breakthrough claim.
