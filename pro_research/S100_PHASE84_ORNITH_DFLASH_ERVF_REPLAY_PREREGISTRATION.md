# S100 Phase84D — DFlash-candidate-workload MoE/transport stress test

## Frozen question

Can the custom Ornith router, LRU52 expert cache, real NVFP4 expert weights,
segmented host-to-device transport and route-adaptive expert kernels execute
the H4 hidden/route workload induced by DFlash candidate blocks?

This experiment is deliberately narrower than a complete decoder.  The target
attention/recurrent states and `attn_post_norm` rows come from the instrumented
llama.cpp run; all router, cache, transport, shared-expert and routed-expert
work after that boundary is executed by the custom ERVF path.

## Inputs

- Pottokao Ornith-1.5 NVFP4 Hugging Face snapshot.
- The committed real target+DFlash callback trace from Phase76.
- The last callback groups aligned to `target_batches`; earlier groups warm the
  per-layer LRU52 cache exactly as prompt prefill would.

## Primary gates

1. Every custom router top-8 row is exactly equal to the authoritative target
   route row in the callback trace.
2. Every H4 block contains 32 routed assignments and has no uncovered expert
   after staging its true misses.
3. Two fresh-cache executions produce bit-identical finite MoE branch outputs.
4. Every copied expert record has the frozen 1,769,472-byte six-segment layout.
5. Cache metadata after every block is identical between planning and replay.

## Claim boundary

Passing Phase84D validates a **DFlash-candidate-workload MoE/transport stress
test**. It does not claim a complete verifier, complete custom target decoder,
generated-text parity, acceptance, or output tok/s. Target attention/recurrent
state is replayed from a trace rather than produced in this process.
