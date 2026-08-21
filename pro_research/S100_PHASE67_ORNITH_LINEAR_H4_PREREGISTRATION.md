# S100 Phase67 — Ornith linear-attention H4 preregistration

Date: 2026-08-21

## Question

Can the Qwen3.5 Gated DeltaNet decode core used by Official Ornith-1.5 and the
Pottokao abliterated checkpoint fit inside the residual Phase66 65 tok/s
budget when four speculative positions are evaluated together?

## Frozen implementation

The measured H4 path consists of exactly three ordered CUDA launches:

1. one fused BF16 `in_proj_a` + `in_proj_b` M4 projection and gate kernel;
2. one depthwise causal-convolution H4 kernel that updates the four-value
   convolution state in place and applies SiLU;
3. one 32-CTA recurrent gated-delta kernel that performs the four token
   updates in sequence and fuses the final per-head RMSNorm + SiLU(z) gate.

The recurrence is float32, matching `mamba_ssm_dtype=float32`. Q and K are
L2-normalized with epsilon 1e-6, Q is additionally scaled by 1/sqrt(128), and
the state layout is `[32, 128, 128]`.

## Inputs

- Real layer-20 auxiliary tensors from both local checkpoints:
  `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, `conv1d.weight`, and
  `norm.weight`.
- Deterministic synthetic float32 H4 hidden, QKV/Z projection outputs,
  convolution state, and recurrent state. The large FP8 projections are not
  timed here because Phase58 already measured them and Phase66 already counts
  their cost.
- Independent NumPy reference implementing the published Qwen3.5 equations.

## Frozen measurements

- 15 warmups and 51 event-timed repetitions per checkpoint.
- Report the median for every launch and the complete ordered path.
- Report projected cost over all 30 Ornith linear-attention layers.
- Audit registers, static shared memory, local memory, and maximum threads.

## Gates

1. GPU output, final convolution state, and final recurrent state each have
   NRMSE <= 5e-5 versus the independent reference and all values are finite.
2. Official and Pottokao complete-path medians are each <= 0.20 ms per layer;
   their 30-layer projection is therefore <= 6.0 ms/H4.
3. Adding the worse 30-layer projection to the Phase66 known-hot floor
   (51.381535 ms/H4) remains below 61.538462 ms/H4 (65 tok/s equivalent).
4. Every kernel uses zero local memory; the recurrent kernel uses <= 96
   registers and supports at least 128 threads per block.
5. The complete path is deterministic for a fixed fresh state within the
   stated numerical tolerance.

Any failed gate is recorded as `measured_fail`; build/runtime problems are
recorded as `technical_failure`. No threshold may be changed after timing.
