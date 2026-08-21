# S100 Phase62: Ornith bulk cold-UVA preregistration

## Question

For 1, 4, 8, 16 or 32 unique complete Ornith expert misses in one layer/H4,
should the runtime execute the real NVFP4 bytes directly from a bounded mapped
pinned ring, or copy the full records to a transient device mirror first?

## Frozen design

- Real Pottokao layer-20 experts starting at expert zero.
- Same Phase59 M1 bulk kernel and arithmetic for hot, direct-UVA and staged
  arms; only source memory differs.
- Each group curve rotates over enough identical pinned record sets to exceed
  four times the measured L2 capacity.
- Direct arm reads codes/scales from the selected mapped-pinned set.
- Staged arm asynchronously copies all selected codes/scales and then runs the
  device kernel. Global scales remain resident in both arms.
- Timings include gate, up, SwiGLU, down and all required transport.

## Gates

1. Every rotating working set is at least 4x L2.
2. Direct and staged outputs match hot bit-for-bit, repeat exactly and are
   finite.
3. Direct-UVA is no slower than staging at group counts 1 and 32.
4. Direct-UVA at 32 groups is below 5 ms/layer.

The measured curve is then used to derive cache-hit requirements. It is not a
full decoder or route-distribution measurement.
