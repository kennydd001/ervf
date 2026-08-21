# S100 Phase68 — Ornith full-attention H4 preregistration

Date: 2026-08-21

## Question

Can Ornith's ten Qwen3.5 full-attention layers fit inside the Phase67 residual
budget at the two previously adjudicated serving contexts, 128 and 1024?

## Frozen semantics

- 16 query heads, 2 KV heads, head dimension 256, GQA ratio 8.
- `q_proj` is interpreted as 16 contiguous `[query256, gate256]` pairs.
- Real layer-23 BF16 Q/K norm weights are used with Qwen3.5's `(1 + weight)`
  convention and epsilon 1e-6.
- The first 64 dimensions receive partial RoPE with theta 10,000,000.
- Four current K/V rows are appended before attention; H4 row `t` sees the
  prior context plus current rows `0..t`.
- Attention softmax and cache are float32 for this custom-runtime experiment.
  The four FP8 Q/K/V/O projections are excluded because Phase58/66 already
  time and budget them.
- The raw query gate is applied as `attention_output * sigmoid(gate)` before
  the already-budgeted O projection.

## Frozen arms

One preparation/append kernel is shared by three attention kernels:

- G1: one query head per CTA;
- G4: four query heads in the same KV group per CTA;
- G8: all eight query heads in the same KV group per CTA.

For each context, the production dispatch selects the lowest complete-path
median among arms that pass the correctness and resource gates. No threshold
or arm is added after timing.

## Measurements

- Official and Pottokao layer-23 norm tensors.
- Contexts 128 and 1024, deterministic synthetic projection outputs and prior
  K/V cache.
- Independent NumPy causal-attention reference.
- 15 warmups and 51 CUDA-event repetitions.

## Gates

1. Prepared Q, appended K/V, and every arm's gated attention output have NRMSE
   <= 5e-5 versus the independent reference and are finite.
2. Fresh-state outputs are bit-deterministic per arm.
3. The selected complete path is <= 0.40 ms/layer at both contexts, so ten
   full-attention layers cost <= 4.0 ms/H4.
4. Adding the worse selected ten-layer cost to the Phase67 known floor
   (53.346655 ms/H4) remains below 61.538462 ms/H4.
5. All kernels use zero local memory, <= 96 registers, and support the selected
   256-thread block.

Any failed gate is `measured_fail`; compilation/runtime failure is
`technical_failure`.
