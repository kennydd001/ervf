# S100 Phase 13C — temporal activation-delta census

Date: 2026-08-18

The same checkpoint was run on the same 10 `_01` calibration and 10 `_02`
validation prompts, with 64 generated tokens per prompt. The census measured
`x_t-x_(t-1)` cosine, norm ratio, coordinate top-k energy for 32–512
components, and per-vector symmetric int8/int4 reconstruction error for the
Mamba, attention, routed-MoE and final-norm activation families.

This is a component screen only. The preregistered decision metric is output
energy retained by sparse-column `W delta`; coordinate energy is not a valid
substitute for that metric. No sparse-column implementation, token-quality gate
or speed measurement was opened, so promotion remains **false**.

As a useful negative screen, validation top-256 coordinate energy was about
45% for MoE input, 66% for Mamba input, 76% for Mamba output and 68% for
attention input. These values are far below the 99% output-energy target even
before the missing matrix projection is considered.
