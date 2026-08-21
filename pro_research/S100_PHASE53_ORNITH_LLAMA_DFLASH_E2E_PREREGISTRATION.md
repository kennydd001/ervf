# S100 Phase53 — independent llama.cpp Ornith DFlash E2E

## Question

Does the exact Pottokao Ornith-1.5 NVFP4 + DFlash GGUF pair produce correct
greedy speculative output and a net end-to-end speedup on the 8 GiB RTX PRO
2000 laptop when target GPU placement is held fixed?

## Frozen artifacts and runtime

- Target repository revision:
  `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash-GGUF@09cec755dab944bddc60bc068ae01bd75271dae8`.
- Target: `Ornith-1.5-35B-A3B-abliterated-NVFP4.gguf`.
- Draft: `dflash-draft-Ornith15.gguf`.
- Official llama.cpp Windows CUDA 13.3 release build 10549, commit `b2e5e9b28`;
  this is the release variant capable of carrying SM120/Blackwell kernels.
- Target placement: exactly 10 GPU layers in both arms, fit disabled.
- Draft placement: all draft layers on GPU in the DFlash arm.
- Context 4096, flash attention on, one server slot, DFlash K=8.
- One unmeasured warm-up followed by fixed coding and arithmetic prompts, 64
  greedy output tokens maximum each.

## Arms

1. `baseline`: target only.
2. `dflash_k8`: identical target arguments plus the DFlash draft.

## Gates

1. Both arms load and serve every request without technical failure.
2. Every measured completion is non-empty and reports at least one output token.
3. Greedy completion text is byte-identical between baseline and DFlash for
   each prompt.
4. DFlash execution exposes acceptance/drafting evidence in server logs or
   metrics and accepts at least one drafted token.
5. Geometric-mean request tok/s for DFlash is greater than baseline.

## Claim boundary

This is an independent two-prompt smoke, not a representative quality or
Spec-Bench result. The fixed 10-layer hybrid CPU/GPU placement is deliberately
slower than a multi-GPU all-resident deployment. A red speed gate establishes
the local break-even direction but does not invalidate the drafter or the
custom Phase49/52 runtime path.
