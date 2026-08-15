# PORT80B-T0-R — official layer-0 reference and natural-route preregistration

Date: 2026-08-13  
Phase: CPU reference/capture only  
Target: `Qwen/Qwen3-Coder-Next` at revision
`a19358a7659bd1f564300250ee189120c49a562f`

## Question and claim

T0-R asks whether a pinned official reference can execute the complete first
decoder layer from real token embeddings and real BF16 layer-0 weights, then
produce provenance-locked layer-0 hidden states, natural 512-way router logits,
top-10 expert IDs/weights, real-weight Q5 reference outputs, and full composed
layer outputs.

A pass is a **real-weight, natural layer-0 route/reference capture**. It is not
a full-depth model-quality, LM-logit, decode-performance, or industrial claim.
Layer 0 has no access to the final norm/LM head, which are outside shard 1;
therefore calling any T0-R output a language-model logit is forbidden.

## Immutable upstream inputs

- Official safetensors index: 6,759,619 bytes, SHA-256
  `e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b`.
- Required official shard: `model-00001-of-00040.safetensors`, exactly
  3,999,619,288 bytes, official SHA-256
  `8e9a5171...` (the full 64-hex digest must replace this abbreviated metadata
  before preflight may pass).
- The shard must contain all 1,550 indexed `model.layers.0.*` tensors plus
  `model.embed_tokens.weight`; no tensor may be sourced from another model.
- Reference environment: `.venv-next-ref`, Python executable and package
  paths locked at preflight; `transformers==5.15.0`, `torch==2.12.1+cu132`.
- Reference sources include the installed generated Qwen3-Next modeling and
  configuration modules. Their full SHA-256 hashes, plus all imported
  Qwen3-Next kernel/fallback modules used at runtime, must be written into the
  reference lock before execution.
- Architecture cross-check: local llama.cpp commit
  `9558fa44c92746a58dd07ad1bf0c889715b938a6`; its `qwen3next.cpp` SHA-256 is
  `651c74364d25a65be5d3b96fb5f9ff1675849a3970fdbf545cfdccac87bb23ab`
  and CUDA GDN source SHA-256 is
  `6c95caa9dff67279b23b39058a74ddb4ab6d634f651716f82482e06e53f027d8`.
  This is a semantic cross-check, not the primary reference.

No download is authorized by this document. Metadata, complete URLs, exact
64-hex LFS hashes, config/tokenizer hashes, runner hash, and this preregistration
hash must be frozen in a CPU preflight before a separate download go.

## Fresh deterministic prompt lock

Freeze inputs before any result exists. Use four domains and one prompt per
domain: general, Python code, mathematics, and multilingual. Each prompt is a
fixed UTF-8 literal stored in a prompt-lock JSON. Tokenize with the pinned
official tokenizer using `add_special_tokens=False`; take exactly the first 16
tokens and reject rather than pad if fewer than 16 are produced. Store prompt
UTF-8 SHA-256, tokenizer-file hashes, token IDs, and token-ID byte hashes.

The prompts must be fresh: their exact token-ID arrays may not equal any
existing P0/P0C/P4D decision row. T0-R uses the complete 16-token prefix for
each prompt. The primary numerical/route rows are positions 8–15 (32 held-out
positions total); positions 0–7 are disclosed calibration/reference-debug
rows and may not be used for threshold tuning.

## Frozen execution and arithmetic

1. Map shard 1 read-only and verify the complete file SHA before reading
   tensors. Verify all 1,550 layer-0 key names, dtypes, shapes and payload
   offsets against the pinned index/N4A contract.
2. Instantiate only embedding plus the exact official layer-0 graph. CPU is
   the primary reference device; CUDA must remain uninitialized. Force
   `torch.set_num_threads(1)`, deterministic algorithms, no autocast, no
   remote kernels, no `trust_remote_code`, and no network.
3. Execute the full layer-0 forward: real embedding, input RMSNorm, complete
   layer-0 Gated-DeltaNet including conv/recurrent state, attention residual,
   post-attention RMSNorm, real 512-way router, normalized top-10 routed
   mixture, sigmoid-gated shared expert, and final layer residual.
4. Preserve the reference BF16 path and a separately implemented selective
   dequantized-Q5 expert path. Q5 is symmetric group-128 RTN: FP32
   `max(abs(group))/15`, round-to-nearest-even, clamp `[-15,15]`, signed
   5-bit two's-complement packing in the frozen STREAMQ5 order, BF16 stored
   scales, dequantized product rounded using the candidate's frozen semantics.
5. Quantize all 512 routed experts plus the shared expert to make the later
   physical bank route-independent. Never copy the old uniform PORT80B
   payload. Every record binds source tensor key/SHA, codes SHA, scale SHA,
   payload CRC, layer=0, expert ID, projection and dimensions.

## Raw outputs that must be retained

Use one non-lossy `.safetensors` artifact per prompt, plus a JSON manifest.
For all 16 positions retain:

- token IDs and embedding output;
- input-norm output;
- GDN output, conv state and recurrent state after each position;
- post-attention residual and post-attention norm;
- all 512 FP32 router logits;
- top-10 IDs and normalized FP32 top-10 weights;
- each of ten routed BF16 expert outputs and their weighted sum;
- shared expert raw output, sigmoid gate and gated output;
- BF16 composed layer-0 output;
- dequantized-Q5 routed outputs, shared output and composed Q5 output.

Each tensor row stores shape, dtype, byte count and SHA-256. Raw bytes—not
only digests—must remain available so an independent verifier can recompute
every digest and metric. Preserve a second process's replay artifacts under a
separate filename.

## Hard gates

All are conjunctive and frozen:

1. Exact revision/index/shard/config/tokenizer/reference/runner hashes; shard
   size exactly 3,999,619,288; no missing/unexpected layer-0 tensors.
2. CPU-only proof: CUDA never initialized, no GPU allocation/kernel, no
   network, shard mapped read-only.
3. Exactly four prompts × 16 tokens; all input and output tensors finite;
   exactly 64 router rows of 512 finite FP32 logits.
4. Every route row has ten unique IDs in `[0,511]`; stored weights are positive,
   finite, non-increasing in rank and sum to `1 ± 2e-6`. Recomputing FP32
   softmax/top-k from stored logits yields identical IDs and weights within
   `2e-6` absolute error.
5. Two clean process executions are bitwise identical for token IDs, route IDs,
   Q5 codes/scales/records, and all BF16/raw state/output tensors. FP32 router
   logits and route weights must also be bitwise identical. No median-based or
   tolerance-only determinism substitution is allowed.
6. Every Q5 record independently dequantizes to the frozen quantized tensor;
   all record/source identity and payload CRC checks pass. At least 95% of the
   513 expert-record payload SHA-256 values are unique, and no two distinct
   expert IDs may have identical triples of projection hashes unless their
   source BF16 triples are also byte-identical and reported explicitly.
7. Q5 versus BF16 quality on the 32 held-out positions: composed layer output
   relative L2 ≤ 0.08, cosine similarity ≥ 0.995, every-domain relative L2
   ≤ 0.12, and top-10 router IDs remain exact because routing is computed from
   the common real BF16 post-attention-normalized input.
8. Negative controls on at least eight held-out route rows: replacing rank 0
   with a different expert, swapping ranks 0/1 without swapping weights, and
   substituting a hot/cold-boundary expert each change the composed Q5 output
   byte digest and yield at least one differing BF16 word.
9. Available RAM ≥ 8 GiB before execution, ≥ 2 GiB at every checkpoint, peak
   RSS ≤ 12 GiB, no exception, and every mmap/file handle closes cleanly.

Failure closes T0-P. No retuning after held-out outputs are visible.

## Explicit non-claims

T0-R does not prove final vocabulary logits, next-token cross-entropy,
full-depth quality, natural routes beyond layer 0, physical 499+13 transport,
latency, throughput, endurance, or an LLM breakthrough.

