# PORT80B-T0Q5-R2 — real layer-0 Q5 numerical-quality preregistration

Date: 2026-08-13  
State: immutable design candidate; **no runner/model/forward/bank until independent source GO**  
Supersedes: T0Q5-R1 (immutable NO-GO design evidence)  
Checkpoint: `Qwen/Qwen3-Coder-Next@a19358a7659bd1f564300250ee189120c49a562f`  
Shard 1: `3,999,619,288` bytes; SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`

## Exact claim

Can a source-derived Q5 reconstruction preserve the official Qwen3-Coder-Next layer-0 expert-plane numerics on fresh natural whole-sequence inputs? A pass proves only: (a) codec/source reconstruction for all 512 routed experts plus shared, and (b) numerical quality for the expert records actually selected by the frozen 32 natural held-out rows plus shared. It does not prove numerical quality for all 512 experts, full-model logits/quality, GPU/DirectPath correctness, throughput, endurance or deployment.

## Fresh/disjoint input lock

Use only the frozen R1 generator `generate_port80b_t0q5r1_prompts.py`, seed `PORT80B-T0Q5R1-FRESH-DISJOINT-2026-08-13-v1`, four fixed natural domains and no rejection/filtering. Before any shard/model touch, canonical generation must yield exactly four rows and **exactly 16 IDs per row**. Against every T0-R12 row, enforce all `4×4` pairs: UTF-8 texts unequal, token-ID sequences unequal, and token-byte SHA-256 unequal. Also enforce pairwise inequality inside the new set. Freeze canonical bytes and tokenizer/source hashes before reference outputs. Positions 0–7 are calibration; primary rows are all four prompts at positions 8–15 (32 causal rows), with no route-based selection.

## Reference and graph-matched arms

One official CPU whole-sequence length-16 forward per prompt, fresh cache, pinned BF16/runtime/source contract. Retain the whole `[1,16,*]` official layer output and MLP input; direct native-BF16 router logits/weights/int64 IDs; routed expert aggregate; shared raw, sigmoid gate and gated shared; complete MLP output; final cache. Direct gate tuple must equal a diagnostic second call.

For every prompt run two manual full-length-16 arms with the same captured IDs/weights and increasing expert-ID dispatch:

1. `source_bf16_graph`: official BF16 source weights, one fused `gate_up` matrix/call per routed expert over exactly the token rows assigned to that expert, split gate-then-up, official SiLU and BF16 cast points, BF16 down, official increasing-ID `index_add_`; shared uses its official gate/up/down graph over all 16 rows and the captured shared sigmoid gate.
2. `q5_graph`: identical batches, dispatch, fusion, operations, order and cast points, replacing each source matrix only by its decoded Q5 BF16 matrix.

The graph control must match the official captured routed aggregate, shared raw/gated and complete MLP output bitwise; maximum one BF16 ULP is permitted only if the exact differing tensor/words are retained and the independent verifier attributes it to a frozen documented Torch primitive. Otherwise the outcome is `graph_control_negative`, and Q5-vs-official metrics cannot pass. This separates quantization loss (`q5_graph` versus `source_bf16_graph`) from backend/shape/fusion loss (`source_bf16_graph` versus official).

Both manual arms execute full 16-token prompt contexts; only positions 8–15 are scored. A row-at-a-time Q5 GEMV is not the primary quality oracle.

## Exact 64-byte wire, payload and manifest

The R2 codec source defines `HEADER_FORMAT = <4sHHHBBIIH2xIII28s`:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | magic `SQ5M` |
| 4 | 2 | version `1` |
| 6 | 2 | layer `0` |
| 8 | 2 | expert identity `0..512` (`512` = shared) |
| 10 | 1 | projection `0=gate,1=up,2=down` |
| 11 | 1 | bits `5` |
| 12 | 4 | rows, little-endian |
| 16 | 4 | columns, little-endian |
| 20 | 2 | group `128` |
| 22 | 2 | zero reserved |
| 24 | 4 | code bytes `655,360` |
| 28 | 4 | scale bytes `16,384` |
| 32 | 4 | CRC32 of `codes || scales`, zlib chaining |
| 36 | 28 | zero reserved |

Then codes, scales and exactly 4,032 zero padding bytes. Each matrix is exactly 675,840 bytes; each expert triple is 2,027,520 bytes; expert triples 0–511 followed by shared 512 total exactly 1,040,117,760 bytes. Manifest rows bind revision, layer, shared flag, expert, projection/name, shape, byte offsets/lengths, source-tensor key/dtype/shape/SHA-256, header/codes/scales/padding/record SHA-256 and CRC32. The top-level manifest binds ordered record-manifest SHA-256, exact bank SHA-256/bytes and codec-source SHA-256.

Quantization is symmetric group-128 RTN: nonzero scale `max(abs(group))/15`; zero scale exactly BF16 `1.0` and q all zero; `q=clip(round-ties-to-even(w/scale),-15,15)`; stored field `q+15` is only 0..30; 31 forbidden; eight fields are little-order in one 40-bit word; scales are little-endian BF16; decoded weights `(field-15)*scale` are BF16 before graph execution. CRC is initialized by `zlib.crc32(codes)` then continued with scales. Header and all padding are independently parsed and exact.

Build streams one tensor/record at a time to create-new `.inprogress`, fsyncs, independently reparses and reconstructs all 1,539 records, then promotes with a recoverable manifest/commit protocol. No dequantized or duplicate bank is retained. All-512 evidence is explicitly codec/source identity only.

## Numerical gates (32 rows, all conjunctive)

Every raw tensor is retained with key/dtype/shape/bytes/SHA-256; every metric is independently recomputed. For routed, shared raw, shared gated, complete MLP and reconstructed layer (`official_layer - official_mlp + candidate_mlp`) report per-row max-abs, rel-L2, cosine, BF16 differing words and max BF16 ULP.

- `source_bf16_graph` versus official: graph-control rule above.
- Q5 routed rel-L2 `<=0.08`.
- Q5 shared-raw rel-L2 `<=0.08`; shared-gated rel-L2 `<=0.08`.
- Q5 complete-MLP rel-L2 `<=0.08`.
- Q5 reconstructed-layer rel-L2 `<=0.02`, cosine `>=0.999`, max-abs `<=0.125`.
- Across 32 reconstructed-layer rows, mean rel-L2 `<=0.01`; no percentile substitution.

All IDs/weights are the official direct tuple, finite, unique/in-range, positive/non-increasing; every retained array is finite. Absence of natural cold-tail routes is reported honestly.

## Deterministic negative controls

Fixed control rows are prompt 0–3 at positions 8 and 15. Each safe path must reject before normal output. Each unsafe bypass is separately labelled and must change at least one complete-MLP BF16 word; failure to change is a control negative, never permission to select another row/mutation.

1. `wrong_expert`: replace rank-0 expert with the smallest ID in `0..511` absent from that row, retain the original weight; header/manifest requested identity remains original and must reject.
2. `fixed_boundary_identity`: request expert 499 but provide record 498, independent of natural routing; execute its unsafe bypass on the fixed row using the row's rank-0 weight. This is synthetic integrity evidence, never natural-cold quality.
3. `projection_swap`: for the row's rank-0 expert, request gate projection 0 but supply its up projection 1; header shape may match but projection identity must reject.
4. `code_mutation`: shared-down record, choose the lexicographically first code field whose decoded q is not zero and whose corresponding fixed shared activation element is nonzero; flip q one step toward zero while staying 0..30, do not update CRC/hash. CRC/hash must reject. The frozen selection algorithm may inspect source/input values but not mutated output. If no such field exists, the experiment is blocked rather than retuned.

## Resources, outcomes and next step

Require start available RAM `>=16 GiB`, Windows peak working set `<=12 GiB`, minimum available RAM `>=2 GiB`, start free disk `>=4 GiB`, total newly retained artifacts `<=1.10 GiB`, CUDA uninitialized, no hidden weight/full-bank copy and complete cleanup.

Frozen outcomes: `real_layer0_q5_numerical_quality_pass`, `real_layer0_q5_quality_negative`, `graph_control_negative`, `codec_or_identity_negative`, `blocked`, `invalid`. A pass opens only a new preregistered physical mapped/pageable transport gate using the same immutable bank.
