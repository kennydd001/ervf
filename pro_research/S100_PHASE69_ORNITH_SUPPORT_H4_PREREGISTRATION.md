# S100 Phase69 — Ornith router/norm/reduction H4 preregistration

Date: 2026-08-21

## Question

Do the remaining all-hot layer operations fit below 65 tok/s after the measured
Phase67 linear core and Phase68 full-attention core are included?

## Frozen path

For one H4 layer support step:

1. fuse attention residual addition with post-attention Qwen3.5 RMSNorm;
2. compute the real BF16 256x2048 router and 1x2048 shared gate together in
   direct-L2 M4 form;
3. serially select top-8 per token on device, normalize selected exponentials,
   and map all-hot expert IDs to cache slots in the same launch;
4. fuse eight route-order weighted expert reductions, sigmoid-gated shared
   output, residual addition, and the next input/final RMSNorm.

One initial input RMSNorm precedes 40 repeated support steps. This is 81 real
H4 norm reductions and includes launch/orchestration time for all support
kernels. The top-8 normalization uses the algebraically equivalent identity
`softmax(selected) / sum(softmax(selected)) = exp(selected-max) /
sum(exp(selected-max))`.

## Inputs

- Real layer-20 input/post norms, router and shared-gate weights from Official
  and Pottokao.
- Deterministic synthetic attention branches, 32 route-order expert outputs,
  shared output and residual stream.
- Identity all-hot `slot_of[256]`; cache misses and transport remain governed
  by Phase62 and real route traces.

## Conservative indirection correction

Phase66 budgets Phase59's contiguous 32-assignment bulk kernel. A production
cache consumes indirect slots. Add, over all 40 layers, the positive absolute
difference between Phase60 M1 indirect and Phase59 bulk32. This correction is
kept even though the measurements were collected in separate sessions.

## Measurements and gates

- 15 warmups and 51 event-timed repetitions.
- Independent NumPy one-step reference.
- Normed streams, router/shared logits and combined residual/norm output:
  NRMSE <= 5e-5 and finite.
- Top-8 IDs exactly match the reference; route weights have NRMSE <= 5e-5;
  all slots hit and equal their expert IDs.
- Fresh-state support output is bit-deterministic.
- Forty support steps plus the initial norm cost <= 3.0 ms/H4.
- Phase68 floor + support cost + conservative indirect correction remains
  below 61.538462 ms/H4.
- Every kernel has zero local memory, <= 96 registers and supports its launch
  block size.

Failed gates produce `measured_fail`; compile/runtime errors produce
`technical_failure`.
