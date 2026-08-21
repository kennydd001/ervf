# S100 Phase49 — Ornith route-adaptive NVFP4 expert family

## Question

Can exact-size M2 through M8 NVFP4 kernels remove Phase48's padded-H8
break-even penalty for routed Ornith SwiGLU experts?

## Frozen input and control

- The same pinned Pottokao layer-20 expert-0 checkpoint record and deterministic
  eight-row input used by Phase48.
- Multiplicities 2 through 8 are benchmarked independently.
- Each candidate executes gate, up, SiLU multiplication and down with a kernel
  compiled for exactly that multiplicity.
- The control is the production ERVF H1 expert invoked sequentially the same
  number of times.
- All weights are hot-resident. Host transfer remains separately measured by
  Phase48.

## Gates

1. All projection checkpoint contracts remain valid.
2. Every M2 through M8 output is finite and agrees with the independent
   byte-level reference at normalized RMSE <= 0.005, cosine >= 0.9999 and
   normalized max error <= 0.020.
3. Every candidate agrees with its sequential H1 control at normalized RMSE <=
   0.001 and normalized max error <= 0.005.
4. Every candidate is bitwise deterministic on repeat.
5. Every exact-size candidate M2 through M8 is faster than its same-size
   sequential H1 control.
6. The first beneficial grouped route multiplicity is 2.

## Claim boundary

Passing proves a route-adaptive expert-kernel dispatch curve. It does not prove
the frequency of those multiplicities in Ornith. A real target route census is
still required before integrating the curve into a whole-model tok/s estimate.

