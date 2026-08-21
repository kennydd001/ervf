# S100 Phase54 — Ornith quantized greedy reproducibility matrix

## Question

Is the Phase53 arithmetic divergence an unstable run, a DFlash K=8-only
multi-token verification effect, or a reproducible llama.cpp draft-model path
effect on the quantized NVFP4 target?

## Frozen artifacts and runtime

- Exact target and DFlash GGUF pair from
  `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash-GGUF@09cec755dab944bddc60bc068ae01bd75271dae8`.
- Official llama.cpp Windows CUDA 13.3 build 10549, commit `b2e5e9b28`.
- Exactly 10 target GPU layers, fit disabled, all DFlash layers on GPU,
  context 4096, flash attention, and one server slot.
- Phase53 arithmetic prompt, at most 64 output tokens, `temperature=0`,
  `top_k=1`, fixed seed 5300.
- Prompt caching disabled. Every replicate starts a fresh server process to
  exclude persistent target and drafter replay caches.

## Arms

Two fresh-process replicates per arm:

1. `baseline`: target only.
2. `dflash_k1`: DFlash with draft maximum 1.
3. `dflash_k8`: DFlash with draft maximum 8.

## Gates

1. All six fresh-process cells load and serve a non-empty completion.
2. The two baseline completions are byte-identical.
3. The two DFlash K=1 completions are byte-identical.
4. The two DFlash K=8 completions are byte-identical.
5. Both K=1 completions are byte-identical to the baseline consensus.
6. Both K=8 completions are byte-identical to the baseline consensus.
7. Every DFlash cell exposes positive accepted-token evidence.

## Frozen adjudication

- Gates 2–4 green and either gate 5 or 6 red: reproducible speculative-path
  divergence, consistent with llama.cpp issue #25618 on quantized targets.
- Gate 5 green and gate 6 red: multi-token verification geometry is isolated.
- Gate 5 red: even the narrowest draft-model verification path changes the
  quantized target result; K=8 is not the sole cause.
- Any repeatability gate red: nondeterminism remains and the path-effect claim
  is withheld.

This phase diagnoses correctness only. Fresh-process cold request throughput
is recorded but is not a performance gate.
