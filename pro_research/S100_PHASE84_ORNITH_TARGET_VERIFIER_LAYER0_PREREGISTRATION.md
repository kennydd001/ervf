# S100 Phase84 — target-verifier layer-0 integration gate

## Frozen question

Can the custom target path start from real token embeddings and reproduce an
independent HF-ModelOpt dequantized layer-0 reference after input RMSNorm,
correctly scaled FP8 projections, Gated DeltaNet state mutation, output
projection, residual addition and post-attention RMSNorm?

## Method

- Use the first H4 of the committed target-only 64-token llama.cpp trace.
- Initialize convolution and recurrent state to zero.
- Load real Pottokao embedding, norm, FP8 and DeltaNet auxiliary weights.
- Quantize every projection input as `E4M3(x / input_scale)` and restore the
  product with `input_scale * weight_scale`.
- Compare all 8,192 output values with an independent CPU reference using the
  checkpoint's original FP8 weights and static activation scales.
- Record the llama.cpp `attn_post_norm-0` delta separately. Its GGUF converts
  these three projection families to Q8_0 and is therefore not an exact
  activation reference for the original HF-FP8 checkpoint.

## Gates

1. Custom CUDA E4M3 bytes exactly match PyTorch E4M3 on finite test values.
2. Output is finite and repeat-exact from fresh zero state.
3. HF layer-0 NRMSE is at most 1e-4 and max absolute error at most 1e-2.
4. Convolution and recurrent state are actually mutated.

Passing this gate proves only embedding through layer-0 attention/state. It is
not the complete 40-layer verifier and does not claim output tok/s.
