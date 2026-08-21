# S100 Phase84 — authoritative target-H4 MoE/transport discrepancy test

## Frozen question

Does the 40-layer custom router/LRU52/real-H2D/MoE path remain near the
Phase84D 67.376 ms/H4 observation when its H4 blocks come from an authoritative
target/reference sequence rather than a DFlash-candidate workload?

## Frozen method

- Use the committed 64-token target-only Ornith trace captured from llama.cpp.
- Warm each layer's LRU52 with authoritative rows 0..31.
- Measure authoritative rows 32..63 as eight contiguous H4 blocks.
- Execute the identical Phase84D router, cache planner, six-segment NVFP4
  transfer, routed experts and shared expert with the same real weights.
- Compare normalized misses, bytes and wall-clock ms/H4 with Phase84D.

The DFlash trace may be read only after the target-only measurement as a frozen
comparison result. It must not supply hidden states, routes, prefetch choices,
cache mutations or timing to this run.

## Primary gates

1. Custom BF16 router top-8 IDs exactly match all authoritative target rows.
2. Cache planning covers all 32 assignments in every H4 block.
3. Fresh-cache repeats are bit-identical and finite.
4. All misses perform the real 1,769,472-byte expert transfer contract.
5. The run reports measured timing and misses without calibrated waits,
   component substitution, oracle prefetch or DFlash control signals.

## Claim boundary

This experiment isolates the MoE/router/cache/transport discrepancy. It is not
the complete target verifier, does not execute attention or recurrent state,
and does not claim output tok/s.
