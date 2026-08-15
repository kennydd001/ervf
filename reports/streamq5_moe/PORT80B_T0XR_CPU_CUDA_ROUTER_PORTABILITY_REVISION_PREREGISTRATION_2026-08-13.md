# PORT80B-T0X-R CPU/CUDA router-portability revision

Date: 2026-08-13

This revision inherits every hypothesis, input hash, operation, gate, resource limit, no-retry rule, and claim boundary from `PORT80B_T0X_CPU_CUDA_ROUTER_PORTABILITY_PREREGISTRATION_2026-08-13.md`.

The first attempt stopped before CUDA initialization and before creating any output because it required `[1,16,2048]`, while the immutable R6-D raw artifact stores the flattened official gate input as `[16,2048]`. The sole semantic repair is to accept exactly `[16,2048]` BF16. No tensor value, routing operation, threshold, comparison, or adjudication is changed.

The invalid attempt is retained in `port80b_t0x_invalid_pre_execution_attempt.json`. T0X-R uses create-new runner, lock, result, raw-artifact, and verification paths. A second repair or retry after T0X-R opens output is forbidden.
