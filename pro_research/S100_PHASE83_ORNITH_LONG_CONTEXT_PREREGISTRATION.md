# S100 Phase83 — long-context adjudication of the 66.56 tok/s stack

## Question

Does the Phase69 all-hot component stack, measured at 60.0955 ms/H4 or
66.5607 tok/s with ctx1024 full attention, retain useful throughput and fit in
8 GB at real 50k and 100k context?

## Frozen setup

- Pottokao Ornith-1.5 layer-23 BF16 Q/K norms and the exact Phase68 Qwen3.5
  H4 full-attention kernels.
- Base contexts 1,024, 4,096, 16,384, 50,000 and 100,000.
- All existing Phase68 arms (`g1`, `g4`, `g8`) compete independently at each
  context; selection is minimum p50 among reference-correct arms.
- FP32 K/V cache, matching the current custom implementation: two KV heads,
  head width 256, K plus V, ten full-attention layers.
- The non-full-attention portion is frozen as Phase69's 60.095487602-ms floor
  minus Phase68's 10-layer ctx1024 contribution. Each context substitutes its
  newly measured 10-layer contribution.
- 0.5 GiB is the frozen Phase46 runtime/KV reserve. No unimplemented KV
  quantization, host offload or sliding-window assumption is credited.
- Quality is checked against the independent Phase68 NumPy reference at every
  context. Three warmups and eleven GPU-event repetitions per arm.

## Gates

1. Every selected arm is finite, deterministic and has NRMSE <= 5e-5.
2. The ctx1024 recomputation is within 10% of Phase68's frozen contribution.
3. The projected complete component stack remains at least 65 tok/s at 50k.
4. The projected complete component stack remains at least 65 tok/s at 100k.
5. The current FP32 ten-layer K/V allocation fits the 0.5-GiB runtime reserve
   at both 50k and 100k.

This is the strongest executable custom-component test available. It is not a
tokenizer, DFlash acceptance, sampling or full end-to-end decoder claim.
