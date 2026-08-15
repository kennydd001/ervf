# PORT80B-T0-R3 — immutable official layer-0 reference preregistration

Date: 2026-08-13  
Phase: CPU reference and natural-route capture only  
Target: `Qwen/Qwen3-Coder-Next` revision
`a19358a7659bd1f564300250ee189120c49a562f`

This is the immutable state-equivalence strengthening after T0-R2. All older
documents remain evidence. R3 retains R2 provenance, inputs, codec, numerical
quality gates, model scope and claim boundary. It adds only preregistered
official whole-prefix versus official DynamicCache token-step equivalence,
manual-MoE versus official-layer equivalence, and per-step cache-state evidence.
No T0 reference output existed when these stronger gates were frozen.

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
   exposing tensor bytes. Independently parse its safetensors header: exactly
   1,567 indexed tensor entries, all BF16, unique non-overlapping in-range data
   offsets, and names/shapes/dtypes matching the pinned index/configuration.
3. Instantiate only embeddings and decoder layer 0 directly on `meta`, move it
   to CPU BF16 without an intermediate FP32 parameter materialization, then
   load all and only their official tensors. Execute the official layer-0 graph
   from the 16-token
   prefixes: input RMSNorm; complete layer-0 Gated-DeltaNet including causal
   convolution and recurrent state; residual; post-attention RMSNorm; router;
   normalized top-10 routed MoE; sigmoid-gated shared expert; final residual.
4. Preserve the official BF16 reference. Separately quantize every layer-0
   routed expert and shared expert with frozen symmetric group-128 Q5 RTN:
   FP32 `max(abs(group))/15`, round-to-nearest-even, clamp `[-15,15]`, signed
   signed code `q in [-15,15]`, stored field `q+15 in [0,30]`, packed in
   little-order groups of eight five-bit fields in five bytes, and BF16 scales.
   Stored field 31 is forbidden and must be rejected. Independently
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
512 native BF16 router logits, independently recomputed FP32 linear logits,
FP32 softmax probabilities and pre-cast normalized FP32 top-10 weights, official
returned BF16 top-10 weights and IDs, ten BF16 expert
outputs and weighted sum, shared raw/gate/gated output, BF16 layer output,
dequantized-Q5 expert/shared outputs and composed Q5 layer output. Store raw
bytes—not only digests—and per-tensor name, dtype, shape, byte count and SHA.

## Conjunctive hard gates

1. Every provenance item above matches, including complete shard/support/
   environment/prompt/runner/verifier hashes. Exactly the indexed layer-0 and
   embedding tensors are loaded; all payload offsets are valid.
2. CPU-only and offline proof passes; CUDA remains uninitialized and no GPU,
   network, registration or bank-build action occurs.
3. Exactly 4×16 finite rows; native router logits are exactly `[4,16,512]`
   BF16 and the independently recomputed linear logits/probabilities are FP32.
4. Every route has ten unique IDs in `[0,511]`; the pre-cast FP32 normalized
   weights are positive, finite, non-increasing and sum to `1 ± 2e-6`.
   Independently recomputed FP32 softmax/top-k produces identical IDs and
   pre-cast weights within `2e-6` absolute error. Official returned BF16 weights
   must equal the BF16 cast of the recomputed pre-cast weights bit-for-bit; their
   post-cast sum is reported but is not incorrectly gated at FP32 precision.
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

## Additional official-state and decomposition gates

Before any Q5 result is eligible, execute the same four locked sequences by
two official BF16 paths: one whole-prefix forward and one token-at-a-time
forward using a fresh official `DynamicCache(config)` per prompt. The token-step
path is primary because it exercises real decode-state updates. Retain raw
outputs from both paths. On positions 8–15, whole-prefix versus token-step must
have layer-output maximum absolute error at most 0.02, relative L2 at most
`1e-3`, identical top-10 expert IDs, and route-weight maximum absolute error at
most `2e-3`. These frozen tolerances cannot be tuned after outputs are seen.

For every token step retain an explicit cache-state schema and raw tensors:
layer index, state index, `record_past`, `has_previous_state`, conv-state dtype,
shape, byte count and SHA-256, and recurrent-state dtype, shape, byte count and
SHA-256. Expected layer-0 shapes must be derived from the pinned official
configuration/source and frozen by the runner preflight; an absent/uninitialized
state, shape drift or non-finite state closes the gate.

Also execute a manually decomposed BF16 MoE from the official post-attention
normalized input. Fuse per-expert tensors strictly as
`cat([gate_proj, up_proj], dim=0)` and verify source key, BF16 bytes and SHA for
each half before use. Preserve router logits/IDs/weights, per-expert gate/up/
down outputs, shared path, weighted sum and composed output. Router artifacts
must be bitwise identical to the official path; every retained manual BF16 MoE
or composed output must differ from the corresponding official decoder-layer
output by at most one BF16 ULP. The ULP verifier operates on BF16 bit patterns
with sign-aware ordering and reports every differing index.

Any failure closes T0-P3. No post-output retuning is permitted.
