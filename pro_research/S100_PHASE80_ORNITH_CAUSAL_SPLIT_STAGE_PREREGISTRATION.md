# S100 Phase80 — causal same-layer split-stage transport

## Hypothesis

Perfect future routes are unnecessary if each real layer pipelines its
authoritative misses in projection order:

1. after the target router, copy miss gate/up codes and scales;
2. compute all cache-resident routed experts;
3. compute miss gate/up + SwiGLU as soon as those segments arrive;
4. concurrently copy miss down codes/scales and consume them after SwiGLU.

The down projection is causally later than gate/up, creating a second overlap
window inside the same MoE layer.

## Frozen setup

- Real Pottokao layer-20 NVFP4 weights and Phase59 bulk kernels.
- LRU52 miss counts from all 28 warm H4 blocks of the Phase70 real trace.
- 32 unique expert work groups per layer; `32-m` use resident GPU weights and
  `m` use pinned-host real weights copied in the two frozen stages.
- Six exact checkpoint segments; tiny global scales remain resident.
- Pinned rotations exceed 4x L2. One temporary destination set is reused only
  after the previous layer has consumed it.
- Remaining support compute and LM-head time use the same calibrated Phase69
  envelope as Phase73.
- Paired order: compute baseline A, causal candidate, compute baseline B. The
  exposed tail uses the mean of A/B p50s.

## Gates

1. Segment sizes, working set and real-weight contracts pass.
2. Split output is bit-exact against the same copied weights and repeats under
   overlap.
3. Full bulk32 and paired compute envelope stay within 20% and 5% of their
   frozen references.
4. Floor plus paired exposed causal tail is at or below 4000/65 ms/H4.

This is a same-layer mechanism: it makes no route-prediction, DFlash-acceptance
or end-to-end serving claim.
