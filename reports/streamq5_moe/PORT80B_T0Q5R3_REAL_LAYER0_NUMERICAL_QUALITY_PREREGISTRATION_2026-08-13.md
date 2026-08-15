# PORT80B-T0Q5-R3 — official real layer-0 Q5 numerical-quality gate

Date: 2026-08-13  
State: immutable design candidate; execution and runner implementation closed pending independent source GO  
Supersedes immutable R1/R2 NO-GO designs  
Official revision `a19358a7659bd1f564300250ee189120c49a562f`; shard 1 exactly `3,999,619,288` bytes, SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.

## Claim and fresh inputs

A pass proves exact codec/source reconstruction for all 512 layer-0 routed experts plus shared, and numerical quality for only the naturally selected routed records plus shared on 32 frozen held-out rows. It does not prove unselected-expert numerical quality, full-model logits/quality, GPU/DirectPath behavior, speed, endurance or deployment.

Reuse only frozen generator `generate_port80b_t0q5r1_prompts.py`, seed `PORT80B-T0Q5R1-FRESH-DISJOINT-2026-08-13-v1`, four fixed natural domains, one candidate each, no rejection/filtering. Canonical replay before shard/model touch must produce four rows of exactly 16 IDs. For all 16 old-T0-R12/new pairs require unequal UTF-8 text, exact ID sequence and little-u32 token digest; require the same three inequalities for all six new/new pairs. Freeze canonical prompt bytes and tokenizer/source hashes before any output. Positions 0–7 calibrate; all positions 8–15 are primary (32 causal rows), never route-filtered.

## Official capture and exact residual identity

Run one pinned official BF16 CPU whole-length-16 forward per prompt with fresh cache. Clone and retain:

- input embedding and the **actual pre-MLP residual**, captured as the input to `post_attention_layernorm` immediately before that normalization;
- post-attention-normalized MLP input;
- direct official native-BF16 router logits/weights and int64 IDs plus exact diagnostic second-call tuple;
- direct official routed aggregate, shared raw output, shared sigmoid gate, shared gated output, complete MLP output and final layer output;
- final conv/recurrent cache and all runtime/source identities.

Independently verify bitwise:

`official_layer_output == torch.add(actual_pre_mlp_residual_bf16, official_complete_mlp_output_bf16)`

using the frozen native BF16 CPU `torch.add` call, argument order residual first, no `out`, alpha=1, no autocast. This is an eligibility gate.

## Full-shape graph-matched source and Q5 arms

For each prompt, both manual arms operate on all 16 tokens and the same captured official IDs/weights. For every routed expert, gather all assigned token rows in official increasing expert-ID order, call one fused `[1024,2048] gate_up` BF16 linear, split gate then up, apply official SiLU and BF16 multiplication/casts, call BF16 down, then official increasing-ID `index_add_`. Shared executes gate/up/down over all 16 rows and multiplies by the captured BF16 sigmoid gate in official order.

1. `source_bf16_graph` uses re-read official source BF16 matrices.
2. `q5_graph` is identical except matrices are independently decoded to BF16 from the Q5 records.

The source arm must be **strictly bitwise equal** to every captured official routed, shared-raw, shared-gated and complete-MLP tensor. Its candidate layer is computed exactly as:

`source_layer = torch.add(actual_pre_mlp_residual_bf16, source_complete_mlp_bf16)`

and must be bitwise official-layer equal. There is no ULP exception or primitive-specific waiver. Any mismatch is `graph_control_negative`; Q5 quality cannot pass.

The Q5 candidate is exactly:

`q5_layer = torch.add(actual_pre_mlp_residual_bf16, q5_complete_mlp_bf16)`

with the same frozen BF16 call/order. No subtractive reconstruction and no row-at-a-time primary GEMV are permitted.

## Codec, all-1,539 source verification and canonical manifest

Use frozen R3 codec source. Wire header is exactly 64 bytes, format `<4sHHHBBIIH2xIII28s`: magic `SQ5M`, version 1, layer 0, expert 0..512 (512 shared), projection 0/1/2, bits 5, rows, columns, group 128, code bytes 655,360, scale bytes 16,384, CRC32 and zero-reserved bytes. Follow with codes, BF16 scales and 4,032 zero padding. Matrix bytes 675,840; expert triple 2,027,520; exactly 513 triples/1,539 matrices and bank bytes 1,040,117,760.

Quantization: contiguous row-major groups of 128; nonzero scale `max(abs(group))/15`; zero scale exact BF16 1.0 and q=0; round-to-nearest ties-to-even then clip [-15,15]; field=q+15 in 0..30, 31 rejected; eight fields little-order per 40-bit word; scales little-endian BF16; decoded BF16 weight is BF16-cast of `(field-15)*widen(scale_bf16)`.

The builder records source key/dtype/shape/SHA, byte offsets/lengths, header/codes/scales/padding/record SHA and CRC. The **independent verifier** must, for every one of 1,539 records:

1. reopen pinned shard 1 independently and read the source tensor anew by official key;
2. verify BF16 dtype/shape and source bytes SHA independently;
3. independently requantize without calling builder functions and require byte-identical codes and scales;
4. independently parse fields/scales, reject 31, construct the decoded BF16 tensor and require its canonical raw-byte SHA-256 equal the builder-declared decoded-weight digest;
5. independently recompute header, CRC, zero padding, record SHA and exact bank offset.

All-512 evidence is codec/source identity only. Bank build streams one tensor/record at a time to create-new `.inprogress`, fsyncs, verifies, and uses a recoverable promotion; no dequantized/duplicate bank persists.

Manifest core is a JSON object with exact ordered rows (expert ascending, projection 0/1/2), revision/shard/codec hashes, bank bytes/SHA and all record fields. Canonical core bytes are UTF-8 `json.dumps(core, sort_keys=True, separators=(',', ':'), ensure_ascii=False)` with no newline. `manifest_sha256=SHA256(canonical_core_bytes)`. The on-disk envelope is canonical JSON of `{"kind":"port80b_t0q5r3_manifest","manifest":core,"manifest_sha256":...}` followed by exactly one LF. Verifier recreates both canonical core and file bytes; no self-referential hash.

## Exact metric math and gates

Metrics are computed on flattened tensors, reference first. Convert BF16 operands elementwise exactly to IEEE FP64. Let `d=candidate-reference`, with FP64 scalar accumulation in fixed flattened row-major order (not BLAS):

- `max_abs=max_i(abs(d_i))` (0 for empty, though empty is forbidden);
- `ref_l2=sqrt(sum_i reference_i^2)` and `err_l2=sqrt(sum_i d_i^2)`;
- `rel_l2=err_l2/ref_l2`; if `ref_l2==0`, rel-L2 is 0 only when `err_l2==0`, otherwise +infinity/fail;
- `cand_l2=sqrt(sum_i candidate_i^2)` and `dot=sum_i reference_i*candidate_i`;
- cosine=`dot/(ref_l2*cand_l2)`; when both norms zero cosine=1, when exactly one is zero cosine=0;
- differing words=count of unequal raw BF16 uint16 words;
- BF16 ULP: map raw uint16 `u` to signed monotone integer `0x8000-(u&0x7fff)` when sign bit set, else `0x8000+u`; maximum absolute mapped difference.

All sums use a plainly looped FP64 implementation with frozen source hash; NaN/Inf invalid. Aggregate mean is left-to-right FP64 sum of the 32 ordered row rel-L2 values divided by 32.

Score routed, shared raw, shared gated, complete MLP and candidate layer on every primary row. Q5 versus the bitwise-qualified source arm must satisfy: routed rel-L2 <=0.08; shared raw <=0.08; shared gated <=0.08; complete MLP <=0.08; layer rel-L2 <=0.02, cosine >=0.999 and max-abs <=0.125; mean layer rel-L2 <=0.01. Every row must pass; no percentile escape. Report Q5-versus-official too, but because source is bitwise identical these values must equal and are independently checked.

## Deterministic controls

Baseline is each fixed row's unmodified `q5_graph` complete-MLP BF16 bytes. Control rows are `(prompt,position)=(0,8),(0,15),(1,8),(1,15),(2,8),(2,15),(3,8),(3,15)`. The safe identity/integrity path must reject before compute; an explicitly unsafe bypass uses the otherwise identical baseline graph and must differ from baseline by >=1 complete-MLP BF16 word. No alternate row/mutation may be selected after output.

1. Wrong expert: on each control row replace rank-0 by the smallest ID 0..511 absent from that row, keep original rank-0 weight; request original identity while presenting replacement record.
2. Fixed boundary: present record expert 498 for a request of expert 499, independent of natural route. Unsafe bypass replaces rank-0 with 498 and retains its weight; this is synthetic integrity evidence, not natural-cold quality.
3. Projection swap: present rank-0 up record (projection 1) for requested gate (projection 0); unsafe bypass feeds decoded up as gate while leaving actual up unchanged.
4. Code mutation: shared-down record, scan row-major codes and choose first q!=0 whose corresponding shared activation element on the fixed row is nonzero; move q one integer toward zero without updating CRC/hash. If none exists, outcome `blocked`. Unsafe bypass consumes mutated decode.

## Resources and outcomes

Start RAM >=16 GiB; Windows peak working set <=12 GiB; minimum available RAM >=2 GiB; start disk free >=4 GiB; all new retained artifacts <=1.10 GiB; CUDA uninitialized; no hidden full-weight/dequantized copy; cleanup complete.

Outcomes: `real_layer0_q5_numerical_quality_pass`, `real_layer0_q5_quality_negative`, `graph_control_negative`, `codec_or_identity_negative`, `blocked`, `invalid`. A pass opens only a separately preregistered physical transport gate using this immutable bank.
