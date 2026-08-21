# S100 Phase56 — Ornith DFlash lossless ubatch-256 confirmation

## Question

Does the Phase55 `ubatch=256` candidate preserve greedy output and retain a net
speedup across both Phase53 prompts and fresh-process replicates?

## Frozen setup

- Exact Phase53 target/DFlash GGUF pair and official llama.cpp CUDA 13.3 build
  10549.
- Exactly 10 target GPU layers, all DFlash layers on GPU, fit off, context 4096,
  flash attention, one slot, target `ubatch=256`, DFlash K=8.
- Prompt caching disabled at server and request level.
- Coding and arithmetic prompts from Phase53, maximum 64 tokens, temperature 0,
  top-k 1, seed 5300.
- Two fresh-process baseline replicates and two fresh-process DFlash replicates.

## Gates

1. All four processes serve both non-empty completions.
2. Baseline replicates are byte-identical per prompt.
3. DFlash replicates are byte-identical per prompt.
4. Every DFlash completion is byte-identical to the baseline consensus.
5. Both DFlash replicates expose positive accepted-token evidence.
6. Median replicate geometric-mean wall tok/s is higher with DFlash.

This remains a two-prompt local confirmation, not a representative quality or
Spec-Bench claim. The purpose is to establish a safe runnable configuration
while upstream quantized-target draft-model divergence remains unresolved.
