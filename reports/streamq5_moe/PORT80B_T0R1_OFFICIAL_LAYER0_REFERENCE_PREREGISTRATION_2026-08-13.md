# PORT80B-T0-R1 — immutable official layer-0 reference preregistration

Date: 2026-08-13  
Phase: CPU reference and natural-route capture only  
Target: `Qwen/Qwen3-Coder-Next` revision
`a19358a7659bd1f564300250ee189120c49a562f`

This is the immutable correction of the blocked T0-R preregistration. The old
document remains evidence of the blocked attempt. R1 changes provenance only:
it supplies the complete shard digest, official support-file hashes, exact
reference-environment provenance and a result-blind prompt lock. Numerical
thresholds, model scope and claim boundary are not relaxed.

## Falsifiable question and claim boundary

Can a pinned, offline, CPU-only reference execute the complete official first
decoder layer from real embeddings and real BF16 weights, expose its natural
512-way router logits and normalized top-10 routes, and produce independently
replayable BF16 and frozen-Q5 layer outputs?

A pass proves a **real-weight, natural-route, official layer-0 reference**. It
does not prove final vocabulary logits, next-token quality, full-depth routing,
physical mapped-host transport, latency, endurance or an LLM breakthrough.
Layer 0 cannot emit language-model logits because the final norm and LM head
are outside shard 1. Here, “logits” means only the 512 router logits.

## Immutable provenance

- Model revision: `a19358a7659bd1f564300250ee189120c49a562f`.
- Index: 6,759,619 bytes; SHA-256
  `e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b`.
- Shard: `model-00001-of-00040.safetensors`, 3,999,619,288 bytes;
  SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- The index must map exactly 1,550 `model.layers.0.*` tensors and the embedding
  tensor to this shard. No tensor may come from another revision or model.
- Official support files and reference packages are frozen in
  `port80b_t0r1_reference_environment_lock.json`, SHA-256
  `eb31d4e0c1f6a806434ea8a20b6b00200781a89ed9f91e485aad0e3583c0f455`.
- Fresh prompts, UTF-8 hashes, exact token IDs and calibration/held-out split are
  frozen in `port80b_t0r1_prompt_lock.json`, SHA-256
  `f283da7e86adf915431459b08aac967d9c18c3de155699c369f5a55be20e5f34`.
- Primary implementation: Transformers 5.15.0 and Torch 2.12.1+cu132 in the
  resolved environment named in the lock. The inherited base-environment
  paths for Torch, NumPy and safetensors are disclosed, not hidden.
- Architectural cross-check only: llama.cpp commit
  `9558fa44c92746a58dd07ad1bf0c889715b938a6`, `qwen3next.cpp` SHA-256
  `651c74364d25a65be5d3b96fb5f9ff1675849a3970fdbf545cfdccac87bb23ab`,
  and `gated_delta_net.cu` SHA-256
  `6c95caa9dff67279b23b39058a74ddb4ab6d634f651716f82482e06e53f027d8`.

No network, shard download, bank build or GPU action is authorized by this
preregistration. The exact reference runner and independent verifier must be
written, hashed and CPU-preflighted before a separate execution authorization.

## Frozen inputs and partitions

Use the four locked 16-token sequences in this exact order: general, Python,
mathematics, multilingual. Positions 0–7 are disclosed calibration/debug rows.
Positions 8–15 are held out, yielding exactly 32 primary rows. No threshold or
implementation change may use held-out outputs. Tokenization is
`add_special_tokens=False`; the stored IDs must be reproduced byte-for-byte by
the pinned tokenizer before the shard is touched.

The prompt-lock was created before any T0 output existed. Its arrays must not
equal a prior decision row. P4D inputs and routes are from Qwen3-30B-A3B and
are neither natural-route inputs nor evidence for this target.

## Frozen execution

1. Set offline mode before imports; disable remote code and remote kernels.
   Set `CUDA_VISIBLE_DEVICES=-1`, use CPU tensors only, prove
   `torch.cuda.is_initialized()==False` before and after, one Torch thread,
   deterministic algorithms, inference mode, no autocast.
2. Open the official shard read-only and verify size plus full SHA-256 before
   exposing tensor bytes. Validate names, dtypes, shapes and safetensors offsets
   against the index and configuration.
3. Instantiate only embeddings and decoder layer 0. Load all and only their
   official tensors. Execute the official layer-0 graph from the 16-token
   prefixes: input RMSNorm; complete layer-0 Gated-DeltaNet including causal
   convolution and recurrent state; residual; post-attention RMSNorm; router;
   normalized top-10 routed MoE; sigmoid-gated shared expert; final residual.
4. Preserve the official BF16 reference. Separately quantize every layer-0
   routed expert and shared expert with frozen symmetric group-128 Q5 RTN:
   FP32 `max(abs(group))/15`, round-to-nearest-even, clamp `[-15,15]`, signed
   five-bit two's-complement STREAMQ5 packing, BF16 scales. Independently
   dequantize and execute this Q5 path using the same real router inputs.
5. Each of 513 records binds revision, layer, expert/shared identity, exact
   source tensor keys and hashes, codes/scales hashes, byte count and CRC. The
   old 49.9-GB uniform synthetic PORT80B bank is forbidden as numerical input.
6. Run two clean processes from immutable inputs. A second verifier process
   must read raw artifacts, recompute all hashes/routes/metrics and not import
   the candidate physical runner.

## Mandatory raw evidence

Retain non-lossy safetensors plus JSON manifests for both runs. For all 64
positions retain token IDs, embeddings, normalized inputs, GDN output, causal
conv state, recurrent state, residuals, post-attention normalized inputs, all
512 FP32 router logits, top-10 IDs and normalized weights, ten BF16 expert
outputs and weighted sum, shared raw/gate/gated output, BF16 layer output,
dequantized-Q5 expert/shared outputs and composed Q5 layer output. Store raw
bytes—not only digests—and per-tensor name, dtype, shape, byte count and SHA.

## Conjunctive hard gates

1. Every provenance item above matches, including complete shard/support/
   environment/prompt/runner/verifier hashes. Exactly the indexed layer-0 and
   embedding tensors are loaded; all payload offsets are valid.
2. CPU-only and offline proof passes; CUDA remains uninitialized and no GPU,
   network, registration or bank-build action occurs.
3. Exactly 4×16 finite rows; router tensor is exactly `[4,16,512]` FP32.
4. Every route has ten unique IDs in `[0,511]`; weights are positive, finite,
   non-increasing and sum to `1 ± 2e-6`. Recomputed FP32 top-k/normalization
   returns identical IDs and weights within `2e-6` absolute error.
5. Two clean processes are bitwise identical for token IDs, routes, Q5
   codes/scales/records and every retained raw BF16/FP32 state/output tensor.
6. Independent Q5 dequantization equals the frozen quantized tensors. At least
   95% of the 513 expert payload triples are unique; any identical source and
   payload triples across expert IDs are explicitly reported.
7. On the 32 held-out positions, composed Q5-vs-BF16 layer-output relative L2
   is at most 0.08, cosine similarity at least 0.995, and each domain relative
   L2 at most 0.12. Routing stays common and exact.
8. On at least eight held-out rows, wrong rank-0 expert, unsynchronized rank
   0/1 swap, and deterministic expert-498/499 boundary substitution each alter
   the composed-output digest and at least one BF16 word.
9. Available RAM is at least 8 GiB initially and 2 GiB throughout; peak RSS at
   most 12 GiB; all mmaps/handles close; no exception or non-finite value.

Any failure closes T0-P1. No post-output retuning is permitted.

