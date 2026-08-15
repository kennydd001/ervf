# PORT80B T0-R2/T0-P2 independent pre-execution methodology audit

Date: 2026-08-13  
Mode: CPU/read-only inspection; no model forward, CUDA, download, bank build, or registry edit  
Decision: **do not execute immutable T0-R2/T0-P2 as written**

## Executive finding

The official shard and the immutable input locks are real and internally
consistent, but the experiment protocol is not yet executable without either
contradicting its hard gates or producing an oracle whose meaning is
underspecified. T0-R2/T0-P2 must remain immutable evidence. They should be
superseded by T0-R3/T0-P3 with the replacements below; editing R2/P2 in place
would destroy preregistration provenance.

The four principal blockers are:

1. The official router produces **BF16 logits**, not the mandatory FP32 logits
   named in R2. Its normalized top-10 weights are calculated in FP32 and then
   cast back to the router-logit dtype, BF16. Therefore the current
   `1 +/- 2e-6` sum gate cannot be applied to the weights actually consumed by
   the official expert block.
2. A full 16-token official Gated-DeltaNet prefill exposes only the **final**
   convolution and recurrent cache states. It does not expose a state for every
   token position, although R2 requires all 64 per-position states.
3. Instantiating `Qwen3NextDecoderLayer` normally creates FP32 parameters. A
   normal `load_state_dict` copy from the BF16 checkpoint can silently retain
   FP32 module storage, violating both the official-BF16 claim and the 12-GiB
   RSS budget.
4. Bitwise equality between an unspecified native CPU Q5 computation and a
   CUDA Q5 kernel is not a defined correctness oracle. Reduction order, FMA,
   SiLU/sigmoid implementations, intermediate casts, and MoE accumulation
   order are not frozen. Bitwise transport equivalence must instead use the
   same frozen CUDA arithmetic on resident and mapped inputs; CPU-to-CUDA must
   use a predeclared numerical error envelope or an explicitly emulated
   identical reduction tree.

## Inputs inspected

| Item | Observed result |
|---|---|
| T0-R2 preregistration | SHA-256 `0470663d1213ac369ef7096c72f92f7fd6db1d76499cc5add3917e70e6bc647c` |
| T0-P2 preregistration | SHA-256 `d2cea182c685f3d6f40f5e8f4037e54ff240aaa177129b2036562f4102f37483` |
| Current R2/P2 preflight | SHA-256 `930c32fa181637faf946e9c8cf1abca7c318f97b5e7b59a400824c66827ff70a`; recorded 14/14 input checks pass |
| Official revision | `a19358a7659bd1f564300250ee189120c49a562f` |
| Official index | 6,759,619 bytes; SHA-256 `e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b` |
| Official shard 1 | 3,999,619,288 bytes; SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a` |
| Prompt lock | SHA-256 `f283da7e86adf915431459b08aac967d9c18c3de155699c369f5a55be20e5f34`; 4 domains x 16 tokens |
| Environment lock | SHA-256 `eb31d4e0c1f6a806434ea8a20b6b00200781a89ed9f91e485aad0e3583c0f455` |

The current preflight legitimately proves immutable inputs, tokenizer replay,
environment identity, codec sentinels, byte arithmetic, and shard presence. It
does **not** preflight the reference computation or the physical oracle. Its
claim boundary also says no shard payload was read even though its full-file
SHA-256 necessarily reads the payload; that sentence must not be copied into a
future result.

## Independent shard-header audit

The shard is structurally sound:

- safetensors JSON header length: 194,000 bytes;
- payload start: byte 194,008;
- tensors: exactly 1,567, all BF16;
- payload: exactly 3,999,425,280 bytes;
- first relative offset: 0; final relative offset: 3,999,425,280;
- all sorted ranges are adjacent, non-overlapping, and in bounds;
- every shape-derived byte count equals its offset span;
- every key maps to shard 1 in the official index;
- exactly 1,550 keys have prefix `model.layers.0.` and the embedding is also in
  shard 1;
- layer 0 plus embedding contain 3,919,393,152 bytes (3.650219 GiB);
- layer 0 alone contains 3,297,063,296 bytes (3.070629 GiB);
- embedding shape is `[151936, 2048]`, BF16, 622,329,856 bytes.

This clears provenance and byte-layout questions. It does not clear execution
semantics.

## Official Transformers API findings

The locked `modeling_qwen3_next.py` establishes the following behavior.

### State/cache semantics

`Qwen3NextGatedDeltaNet.forward` chooses a prefill/chunk path when there is no
previous state. The chunk rule returns a final recurrent state only when a
cache object is supplied. The cache stores only the final recurrent state and
the last convolution window. A `cache=None` call produces the layer output but
does not preserve either state.

The primary reference must therefore use a **fresh `DynamicCache(config)` for
each prompt**. Reusing one cache across prompts contaminates later prompts with
earlier prompt state. Running the 16 tokens as cached single-token decoding is
not an equivalent capture method: after the first token it selects the
recurrent single-token implementation instead of the chunk-prefill
implementation.

The clean way to satisfy per-position state capture is a frozen prefix ladder:
for each prompt, run fresh-cache prefixes of lengths 1 through 16, retain only
the final state of each prefix, and separately run one primary length-16
prefill. The output at prefix length `k` must equal the primary output at
position `k-1` bitwise. If it does not, report an implementation-compatibility
failure; do not choose whichever path looks better.

The calibration/held-out split is an output-position split, not an independent
context split: positions 8-15 causally depend on positions 0-7. Reports must
state this explicitly.

### Router and MoE arithmetic

The official router performs:

```text
router_logits = F.linear(BF16 hidden, BF16 weight)       # native BF16
router_probs  = softmax(router_logits, dtype=FP32)
values, ids   = topk(router_probs, 10)
values        = values / sum(values)                    # FP32
used_weights  = values.to(router_logits.dtype)          # BF16
```

R2's FP32 router-logit requirement is consequently not the official tensor.
Converting the BF16 logits to FP32 is only an exact widening copy; recomputing
the matrix multiplication in FP32 would be a different router and could change
top-k IDs. Also, the official sparse-expert loop visits experts in increasing
expert-ID order and uses `index_add_`; it does not accumulate routes in rank
order. The BF16 and Q5 paths must freeze and disclose their respective
accumulation orders.

`Qwen3NextSparseMoeBlock.forward` does not return logits or intermediates.
Captures need non-mutating hooks or an explicitly mirrored call graph. Every
hooked tensor must be cloned immediately, because cache tensors are updated in
place.

### Loading semantics

The official checkpoint has 1,536 separate routed-expert projection tensors,
whereas the Transformers layer expects packed parameters:

- `experts.gate_up_proj`: `[512, 1024, 2048]`;
- `experts.down_proj`: `[512, 2048, 512]`.

Transformers maps `qwen3_next` to the `qwen2_moe` conversion and performs
expert-list merging plus gate/up concatenation. A manual partial loader must
freeze that exact expert order and prove every packed expert slice equals the
corresponding source tensor bytes. It must construct the layer on `meta` and
materialize parameters with BF16 assignment. Merely copying BF16 values into a
default layer is not sufficient.

### Unlocked dependency surface

The environment lock names the Qwen3-Next source files, but the actual result
also depends on cache, conversion, kernel dispatch, MoE integration,
activation, and masking sources. R3 must add at least these current hashes:

- `transformers/cache_utils.py`:
  `0d5fd6901ce2b7108eff40e06d7ce29e9b0f9cc8ed2f40d2fb3a2e4d4f43e630`;
- `transformers/conversion_mapping.py`:
  `126a342d8be1942f76207ed30f809b659709c71618ab2b57066bfde19a0a1f83`;
- `transformers/integrations/hub_kernels.py`:
  `50e5b5f938cdb2c5a2f7e90ae1ab3933cb2d505ab38c6c4f4d0226320df4b94a`;
- `transformers/integrations/moe.py`:
  `aee99c71daa326c93d15660a73e4bf76676596e437bf5a6beb6983b2c11e7215`;
- `transformers/activations.py`:
  `5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665`;
- `transformers/masking_utils.py`:
  `e8c497af6979274fc6ae78980ad9893e7850bdb750e46d459f09178123992196`.

`USE_HUB_KERNELS=0` must be set before importing Transformers, in addition to
the offline variables. Record the resolved callable/module for causal
convolution, chunk GDN, recurrent GDN, RMSNorm-gated, and experts. Offline mode
alone does not prove fallback selection when a kernel package or cached
implementation already exists.

## Required T0-R3 replacements

R3 should retain the model, prompts, partitions, quality thresholds, and claim
boundary of R2, but replace its inconsistent execution gates as follows.

1. **Native router evidence.** Retain native BF16 logits with shape
   `[4,16,512]`, their exact FP32 widening, FP32 softmax probabilities, FP32
   top-10 normalized weights before casting, and the native BF16 top-10 weights
   consumed by experts. Never call an FP32 re-matmul “official logits.”
2. **Arithmetic-derived router tolerances.** Let `u32 = 2^-24` and
   `uBF16 = 2^-8`. Gate the FP32 normalized-weight sum at
   `|sum-1| <= 16*u32 = 2^-20 = 9.5367431640625e-7`. Gate the FP32 sum of the
   ten native BF16 used weights at
   `|sum-1| <= uBF16 + 16*u32 = 0.00390720367431640625`. The latter follows
   from at most one BF16 unit roundoff applied to positive normalized weights,
   plus a conservative FP32 normalization/summation allowance. The old `2e-6`
   gate may apply only to the pre-cast FP32 weights, not the BF16 used weights.
3. **Tie stability.** Require every FP32 top-10/top-11 boundary margin to be
   strictly positive and record its minimum. An exact tie is a route-stability
   failure because `topk` tie ordering is not a portable replay contract.
4. **State evidence.** Use fresh-cache length-16 primary prefills and the
   fresh-cache prefix ladder described above. Store per-position convolution
   and recurrent states from the ladder and require bitwise agreement of ladder
   outputs with the corresponding primary outputs.
5. **BF16 materialization.** Instantiate on `meta`, assign BF16 tensors, and
   assert dtype/device/shape for every parameter after strict loading. Verify
   packed expert slices independently against all source tensors.
6. **Deterministic CPU contract.** Before imports/compute freeze
   `USE_HUB_KERNELS=0`, offline variables, `CUDA_VISIBLE_DEVICES=-1`, one Torch
   and interop thread, OMP/MKL thread counts, model `eval()`, inference mode, no
   autocast, deterministic algorithms, float32 matmul precision, MKLDNN enabled
   or disabled choice, process affinity, and denormal policy. Record CPU model,
   Torch CPU capability, all choices, and callable provenance in both clean
   processes. Current observed defaults are 16 Torch threads, 16 interop
   threads, MKLDNN enabled, deterministic mode off, and CPU capability `AVX2`;
   therefore importing the environment is not itself the frozen execution.
7. **Q5 wire and quantizer.** Freeze grouping as contiguous row-major groups of
   128 along the source matrix's last dimension; freeze byte order and scale
   byte order. Specify the zero-group rule. To match the locked P1D builder,
   use scale `1.0` and all-zero codes when `max(abs(group)) == 0`; otherwise
   select codes using FP32 `max/15`, store the scale in BF16, and dequantize
   using the stored BF16 scale. R2 currently leaves the zero-group and
   FP32-versus-rounded-scale order ambiguous.
8. **Accumulation and metrics.** Freeze matrix reduction, activation dtype,
   cast points, expert-ID accumulation order, shared/routed addition order, L2
   denominator-zero behavior, cosine epsilon, and metric accumulation dtype.
   Report native BF16 reference and Q5 numerical quality separately from
   transport exactness.
9. **Controls.** A replacement expert must be distinct, absent from the other
   nine selected IDs where the control requires a unique replacement, and have
   a different verified payload before the output-difference gate is applied.
10. **Replay artifacts.** Give clean runs A/B separate create-new output paths;
    refuse overwrite. Store a canonical tensor manifest with name, semantic
    index, dtype, shape, byte count, raw-byte SHA-256, and parent/source hashes.
    Compare tensor payload bytes, not merely container-file hashes. Freeze and
    hash runner plus independent verifier before either run.

## Required T0-P3 replacements

1. **Separate the two oracles.** Source/decode identity remains CPU-verifiable
   and exact. Physical transport exactness must compare mapped-host execution
   bitwise to a resident-device execution using the same frozen CUDA kernel,
   launch geometry, inputs, reduction order, and cast points. This isolates the
   mapped/pageable mechanism without pretending CPU and CUDA arithmetic are
   inherently bitwise identical.
2. **CPU numerical reference.** If CPU-to-CUDA numerical comparison is kept,
   use a formula fixed before output, not a result-selected scalar tolerance.
   For every FP32 dot of length `n`, compute
   `gamma(2n)=(2n*u32)/(1-2n*u32)` and bound its absolute accumulation error by
   `gamma(2n)*sum(abs(x_i*w_i))`, then add propagated input bounds and
   `0.5*ULP_BF16` at each BF16 cast. Use `n=2048` for gate/up and `n=512` for
   down. Propagate intervals through SiLU with a predeclared derivative bound
   and add one BF16 ULP for the backend transcendental implementation. This
   arithmetic envelope may consume observed magnitudes, but its formula and
   constants must be frozen before results. There is no defensible universal
   fixed ULP count for a complete cancellation-prone MLP.
3. **Do not weaken mechanism exactness.** Resident-versus-mapped arrays remain
   zero-bit-difference. CPU interval compliance is an algorithm check; it is
   not a substitute for the bitwise transport check.
4. **Natural cold routes.** Before physical timing, report whether the locked
   32 natural rows actually select any ID in 499-511. If not, gate 6 may be
   satisfied only as a mechanism-control exercise and must not be reported as
   natural cold-route evidence.
5. **Registration lifecycle.** Count registration as successful only if the
   matching unregister also succeeds. Record all attempts, return codes, API
   errors, and post-cleanup state. A successful register followed by unregister
   failure is a failed lifecycle, not a pass.
6. **Page telemetry boundary.** At 1 Hz, a roughly five-second 320-sample
   component run yields too few telemetry samples for a meaningful p95. Require
   a predeclared minimum post-warm-up telemetry count (recommended 30) or mark
   page rates diagnostic only. Repeated timings after the first touch measure a
   warm OS page cache unless a legitimate, non-destructive cache-state protocol
   is frozen; they must not be called cold-page latency.
7. **Timing integrity.** Keep the exact 320 inclusive timings and 15/30-ms
   gates, but report correctness independently if performance fails. Freeze
   warm-up row order, primary row order, repetition nesting, synchronization,
   host-visibility operation, and whether controls are excluded from timing.
8. **P3 eligibility.** P3 remains closed until an independent R3 verifier passes
   every raw-byte, semantic, resource, and provenance gate. A 14/14 input
   preflight alone does not open it.

## RSS feasibility under the 12-GiB cap

The budget is feasible only with a staged implementation; it is not safe with
a conventional FP32 construction.

| Resident category | Bytes | GiB |
|---|---:|---:|
| Relevant source mapping: embedding + layer 0 | 3,919,393,152 | 3.650219 |
| Materialized BF16 embedding + layer 0 | 3,919,393,152 | 3.650219 |
| Complete Q5 bank | 1,040,117,760 | 0.968685 |
| Conservative simultaneous subtotal | 8,878,904,064 | 8.269124 |
| Rough two-run recurrent/conv/intermediate raw tensors if held | 288,358,400 | 0.268555 |
| Remaining below 12 GiB after both rows above | 3,717,639,424 | 3.462321 |

The recurrent state is `[1,32,128,128]` FP32, 2,097,152 bytes per captured
position; the convolution state is `[1,8192,4]` BF16, 65,536 bytes per captured
position. These dominate neither source nor model memory.

Feasible schedule:

1. hash and validate the shard without retaining copied tensors;
2. meta-construct and BF16-materialize the official layer, execute BF16
   reference, stream raw artifacts to disk, and release temporary hooks;
3. close/delete the materialized model before Q5 replay if only saved
   post-attention inputs, routes, residuals, and reference arrays are needed;
4. quantize records in small batches directly from the read-only source and
   stream them to create-new files; never retain all codes/scales in Python;
5. close the source mapping before mapping the completed Q5 artifact where
   possible;
6. measure `PeakWorkingSetSize` on Windows plus frequent psutil samples; polling
   alone can miss conversion spikes.

An accidental FP32 model would require about 7.300438 GiB for the same weights.
Together with the relevant source mapping and Q5 bank it reaches about 11.919
GiB before Python, Torch, oneDNN reorder/scratch, hooks, or raw outputs and must
be rejected before forward.

Replace the 8-GiB initial-RAM gate with **at least 16 GiB available at process
start**, retain the 2-GiB throughout/post-cleanup reserve, and keep the 12-GiB
peak-RSS cap. Fourteen GiB is the arithmetic minimum implied by a possible
12-GiB process plus a 2-GiB reserve; 16 GiB supplies a fixed 2-GiB margin for
OS/background drift. Also require a pre-forward allocation audit to project no
more than 10.5 GiB steady working set, leaving 1.5 GiB for transient library
scratch under the immutable 12-GiB cap. The current machine had about 50 GiB
available during this audit, so host capacity is not presently the blocker.

## Independent verifier design

The future verifier should be a new, hash-frozen process and must not import the
runner, bank builder, CUDA decoder, or their helpers. It should perform these
conjunctive groups:

1. rehash preregistration, runner, verifier, prompt/environment/dependency
   locks, index, support files, and the full shard;
2. independently parse the safetensors header, validate exact shapes/dtypes,
   contiguous offsets, key-to-shard mapping, and the layer0/embedding key set;
3. prove CPU-only/offline/deterministic runtime settings and resolved fallback
   callables from captured process manifests;
4. prove exact 4 x 16 primary order and 4 x (1..16) prefix-ladder order, fresh
   cache identities, exact sample counts, and no cross-prompt cache reuse;
5. validate every tensor's declared semantic key, dtype, shape, byte count, and
   raw SHA; compare A/B raw tensor bytes and primary-versus-ladder outputs;
6. independently recompute routes from native BF16 logits using the official
   BF16-to-FP32 softmax/top-k/normalization/cast sequence and apply the two
   arithmetic-derived sum bounds plus the top10/top11 margin gate;
7. reconstruct every Q5 code/scale/record directly from source BF16 bytes using
   an independent implementation, including zero groups, field-31 rejection,
   record order, CRCs, and uniqueness;
8. independently recompute R3 quality metrics and controls with frozen
   reduction/cast conventions;
9. for P3, compare resident and mapped GPU evidence bitwise, then independently
   apply the frozen CPU numerical interval formulas; recompute all timings,
   percentiles, page samples, registration/unregistration counts, RAM/VRAM
   minima, and cleanup state;
10. fail closed on missing raw arrays, overwrite/retry evidence, provenance
    drift, unregistered hooks, non-finite values, resource sample gaps, or any
    result written before all cleanup checks completed.

## Authorization boundary

This audit authorizes no execution. The official inputs are adequate for a
real layer-0 truth test, and the host has enough memory, but R2/P2 as written
would produce ambiguous or impossible gates. Execution should wait for frozen,
hashed R3/P3 preregistrations, runner, independent verifier, a no-forward CPU
preflight that checks the replacements above, and separate authorization.
