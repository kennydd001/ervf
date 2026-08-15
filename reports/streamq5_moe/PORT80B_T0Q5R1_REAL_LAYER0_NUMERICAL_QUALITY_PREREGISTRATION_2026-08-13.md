# PORT80B-T0Q5-R1 — real-checkpoint layer-0 Q5 numerical-quality gate

Date: 2026-08-13  
State: preregistered, **closed pending independent source audit**  
Checkpoint: `Qwen/Qwen3-Coder-Next` revision `a19358a7659bd1f564300250ee189120c49a562f`  
Shard 1: exactly `3,999,619,288` bytes, SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`

## Question and claim boundary

On fresh natural 16-token inputs, does a source-exact, differentiated Q5 reconstruction of all 512 routed experts plus the shared expert preserve official BF16 layer-0 numerical behavior on positions 8–15? A pass is only evidence for the **CPU numerical oracle and real layer-0 expert plane**. It is not full-model logits/quality, GPU correctness, DirectPath performance, endurance, tokens/s or an industrial breakthrough.

The verified T0-R12-D2-R3 diagnosis is prior evidence only: official routes and final cache were exact, same-length calls deterministic, and the observed prefix/full difference was shape-dependent BF16 expert arithmetic (expert max-abs `2.44140625e-4`, final-layer max rel-L2 `1.2754e-4`). No D2 prompt, token row, route or output is admitted into T0Q5-R1.

## Frozen fresh input protocol

The separately hashed generator has four new, human-written natural prompts (code review, biology, commercial contract, Dutch infrastructure), one candidate per domain, no rejection/filtering and seed `PORT80B-T0Q5R1-FRESH-DISJOINT-2026-08-13-v1`. Tokenize locally with the pinned official tokenizer, no special tokens, retain exactly the first 16 token IDs. Before any model load, require exact canonical generator replay and prove that every prompt text, token sequence and token-byte digest differs from T0-R12. Positions 0–7 are calibration/debug; the frozen primary set is 32 causal rows at positions 8–15. Routes are captured without selection or filtering; absence of natural IDs 499–511 is reported and cannot be relabelled natural-cold evidence.

## Two immutable phases

1. `reference`: CPU-only official whole-sequence layer-0 forward, fresh `DynamicCache` per prompt. Retain token IDs, embedding, post-attention normalized MLP input, direct official BF16 router logits/weights/int64 IDs, routed-expert output, shared raw output, shared sigmoid gate, exact complete MLP output, layer output and final conv/recurrent cache. The direct gate tuple must equal an immediate diagnostic second call. All tensors are finite and source/runtime identities are retained. No prefix ladder is used: D2-R3 already identified shape-dependent prefix arithmetic and this gate deliberately uses whole-sequence reference and whole-shaped Q5 arithmetic.
2. `q5`: only after independent verification of phase 1. Stream the official shard into exactly 513 differentiated records, then execute the manual Q5 CPU oracle on the frozen 32 primary rows. This phase may read the verified reference raw/result and the official shard, but may not alter them.

No phase overwrites an existing output. Failure writes create-new evidence containing stage, error, runtime/resource state and partial-artifact disposition.

## Exact Q5 wire and bank

- source matrices: each routed/shared expert's `gate_proj`, `up_proj`, `down_proj`, layer 0 only;
- matrix shapes: `[512,2048]`, `[512,2048]`, `[2048,512]` BF16;
- symmetric RTN, contiguous row-major groups of 128 columns;
- nonzero group: `scale=max(abs(group))/15`, `q=clip(round_ties_to_even(w/scale),-15,15)`;
- zero group: `scale=1.0`, all `q=0`;
- stored field is biased `q+15`, therefore only `0..30`; field 31 is forbidden;
- eight fields in one little-order 40-bit/5-byte word; BF16 little-endian scales;
- every decoded value is `(field-15)*scale` and is cast to BF16 before matrix arithmetic;
- record header binds revision, layer 0, expert/shared identity, projection, shape, group size and payload lengths; CRC and SHA-256 bind header/codes/scales/padding and official source tensor bytes.

Each matrix record is exactly `675,840` bytes and each expert triple is `2,027,520` bytes. The bank is exactly `513 * 2,027,520 = 1,040,117,760` bytes (about 0.969 GiB), with experts 0–511 followed by shared. Build by streaming one source tensor/record at a time to a create-new `.inprogress` file, fsync, independently reparse/reconstruct all 1,539 matrix records, then atomic promotion plus a hashed manifest. Peak additional disk including compact raw evidence must remain below 1.10 GiB; no dequantized bank or second bank is retained.

## Frozen numerical oracle

Use the retained whole-sequence BF16 MLP input and official direct route IDs/weights. Dequantize only a selected record at a time. Use native CPU BF16 `F.linear`, official SiLU, BF16 multiplication/casts and official increasing-expert-ID accumulation (`index_add_` semantics), then add `shared_raw_q5 * sigmoid(shared_gate_bf16)` in the official order. Retain raw Q5 gate/up/SwiGLU/down per selected expert-row, routed sum, shared raw/gate/gated, complete MLP output and reconstructed layer output:

`layer_q5 = official_layer_output - official_mlp_output + q5_mlp_output`

This isolates expert-plane quantization while holding the official attention/residual path fixed. The independent verifier reparses/dequantizes the selected bank records and recomputes all metrics. Native CPU arithmetic is the target oracle here; no CPU↔GPU bitwise claim is made.

For each retained array record element count, dtype, shape, byte count and SHA-256. For each primary row report max-abs, relative L2, cosine similarity, BF16 differing-word count and maximum BF16 ULP versus official BF16. Aggregate maxima/minima are derived, never substituted for row evidence.

## Frozen gates

All are conjunctive:

1. Exact checkpoint/tokenizer/config/index/shard/dependency/source hashes and CPU runtime contract; CUDA remains uninitialized.
2. Fresh prompt generator replay and strict disjointness from T0-R12 pass before shard/model touch.
3. Official direct router tuple equals the diagnostic second call; IDs are unique/in-range, weights positive/non-increasing/finite, and all raw tensors finite.
4. Exactly 513 identity-distinct expert records, 1,539 valid matrix headers and exactly `1,040,117,760` bank bytes; every source tensor SHA, reconstruction, CRC, padding and field-range check passes. At least 95% of routed expert triple digests are unique.
5. Exact decoded-source oracle: independent re-quantization produces byte-identical codes/scales for every record; zero/nonzero groups and q+15/field-31 semantics pass.
6. All 32 primary rows execute the exact official captured routes. Q5 routed, shared, complete-MLP and reconstructed-layer arrays are finite.
7. On every primary row: Q5 routed-output rel-L2 `<=0.08`, complete-MLP rel-L2 `<=0.08`, reconstructed-layer rel-L2 `<=0.02`, reconstructed-layer cosine `>=0.999`, and reconstructed-layer max-abs `<=0.125`. These conservative thresholds are frozen before Q5 outputs and must not be retuned.
8. Aggregate reconstructed-layer mean rel-L2 `<=0.01`; all 32 rows must meet the per-row gates (no percentile escape).
9. Four controls on eight fixed primary rows (prompts 0–3 at positions 8 and 15): wrong-expert substitution, 498↔499 hot/cold boundary substitution, gate/up projection swap, and one valid-range code-bit mutation. Identity/header/CRC logic must reject every control before normal execution. A separately labelled unsafe-bypass diagnostic must change at least one composed BF16 word for each control; it is never quality evidence.
10. Available RAM at start `>=16 GiB`, peak working set `<=12 GiB`, minimum available RAM `>=2 GiB`, disk free before build `>=4 GiB`, final added artifacts `<=1.10 GiB`, complete cleanup/close, no GPU or hidden full-weight copy.

## Outcomes

- `real_layer0_q5_numerical_quality_pass`
- `real_layer0_q5_quality_negative`
- `real_layer0_q5_codec_or_identity_negative`
- `blocked`
- `invalid`

A pass opens only a separately preregistered mapped/pageable transport comparison using the same bank and a resident CUDA arithmetic oracle. It does not open full-depth or product claims.
