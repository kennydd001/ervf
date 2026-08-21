# S100 Phase47 — Ornith-1.5 DFlash checkpoint smoke

## Question

Does the DFlash checkpoint embedded by pottokao execute as a deterministic,
finite draft body on the local 8 GiB SM120 GPU, and what is its standalone
block cost at K=8 and K=16?

## Frozen checkpoint contract

- Parent: `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash`
- Draft source: `z-lab/Qwen3.6-35B-A3B-DFlash`, copied unmodified under
  `dflash_draft/`.
- Six BF16 transformer layers, hidden 2048, MLP 6144, 32 query heads, eight KV
  heads, head dimension 128.
- Target residual layers `[1, 6, 11, 16, 22, 27, 32, 37]`.
- Five sliding/causal draft-attention layers followed by one full/non-causal
  draft-attention layer.

## Gates

1. The config and all 69 tensors match the frozen architecture; all checkpoint
   tensors are BF16 and total 385,906,176 parameters.
2. Target-context projection from `[S, 8, 2048]` to `[S, 2048]` is finite.
3. K=8 and K=16 draft blocks both execute and return finite `[K, 2048]`
   normalized hidden states.
4. A repeated launch is bitwise deterministic for each K.
5. Peak Torch allocation stays below 3 GiB, leaving room for the packed target
   head and the future streamed target runtime.

## Claim boundary

This is a real-checkpoint draft-body execution and timing result using synthetic
target residuals and synthetic target embeddings. It is not an acceptance,
quality, target-verification, or end-to-end tokens/second result.

