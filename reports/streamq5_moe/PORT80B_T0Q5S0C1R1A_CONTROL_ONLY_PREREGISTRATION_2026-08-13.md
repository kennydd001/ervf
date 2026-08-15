# PORT80B-T0Q5-S0-C1-R1A — observable shared-down control sentinel

## Status and claim boundary

This is the immutable C1-R1A authorization/lifecycle revision after C1-R1 received scientific/source GO. It changes no arithmetic, evidence, control or threshold. It opens execution with an audit-specific token and moves the valid-commit repeat check before recovery/nonclean adjudication, so a completed immutable bundle returns `already_complete` without mutation. S0-C1-R1A is a CPU-only control-sensitivity validation; it is not a new numerical-quality experiment, held-out test, model/layer pass, performance result, or breakthrough claim. It may reuse the immutable S0-R5 result but shall not rerun or reinterpret its 759-record quality computation.

S0-R5 remains formally verifier-negative because exactly one frozen unsafe-output conjunct failed: the real shared-down Q5 field `(row=0, column=0)` changed `q=6 -> 5`, but at natural `p0/n8` its contribution was smaller than the BF16 output rounding interval and produced zero changed BF16 words. All metadata, digest, decode, retained-array and replay assertions for that arm were correct. S0-C1-R1A tests only whether the same real codec record and checker detect an intentionally observable one-field corruption.

## Immutable provenance

- Official checkpoint: `Qwen/Qwen3-Coder-Next`, revision `a19358a7659bd1f564300250ee189120c49a562f`.
- Shard 1 exact bytes: `3,999,619,288`; SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- S0-R5 raw: `reports/runs/streamq5_moe/port80b_t0q5s0r5_selected_route_validation/s0r5_raw.safetensors`, `1,658,624` bytes, SHA-256 `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`.
- S0-R5 result: `reports/runs/streamq5_moe/port80b_t0q5s0r5_selected_route_validation/s0r5_result.json`, `561,210` bytes, SHA-256 `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91`.
- S0-R5 commit: `reports/runs/streamq5_moe/port80b_t0q5s0r5_selected_route_validation/s0r5_commit.json`, `266` bytes, SHA-256 `d784ded5e7893095e2f27b75695e635c9cc880109736c87496138e3188509372`.
- Independent control diagnosis: `reports/streamq5_moe/PORT80B_T0Q5S0R5_CONTROL_DIAGNOSIS_2026-08-13.md`, SHA-256 `b22808626e45178cb917cebde5aac789ba720d091ef143099948c94f243bf2e0`.
- The S0-R5 verifier result must show every non-control check true and only `controls=false`; C1 must bind its exact bytes/hash if a persisted verifier artifact exists before implementation freeze.

The only source tensor is `model.layers.0.mlp.shared_expert.down_proj.weight`, BF16 shape `[2048,512]`, source SHA-256 `83565fde9bab5de0109f102c0f21cebd6533d7776b8f6fb837400534dbc5e1f5`. Its S0-R5 record evidence is ordinal `758`, codes SHA-256 `7d2311c8c455cb556d7c65b25df833196272c8f195762b9ac3d482afdf68e65d`, scales SHA-256 `85d438d73b626b7513356c4792e947162d18c3115ad08bd2de428d08b47a197b`, combined SHA-256 `ca74f57285f066334ac9adfdf47ea3cc9e3823859b8d7c3c0a775ffb9168f076`, and decoded SHA-256 `9b24af43030dde4854c7a76cdfaf92045f22099f2e198a3f4f78f187a026b91d`.

## Frozen codec and field selection

Independently reread the official BF16 tensor and recreate the S0 codec exactly: symmetric group-128 RTN, `scale = max_abs/15` stored BF16 for a nonzero group, `scale=1` and `q=0` only for an all-zero source group, `q=round(source/scale)` clamped to `[-15,15]`, stored field `q+15` in `[0,30]`, field 31 forbidden, little-order eight fields per five bytes. Independently unpack and require exact q/scales roundtrip plus all hashes above.

Select the first field in row-major `(output_row,input_column)` order satisfying all of:

1. `q != 0`;
2. one step toward zero remains representable: `q' = q-sign(q)` and `q' in [-15,15]`;
3. decoded BF16 weights differ: `BF16(q*scale) != BF16(q'*scale)`.

No output-dependent search or alternate field is permitted. The expected frozen selection is `(row=0,column=0)`, `q=6`, `q'=5`; failure to reproduce it is `invalid_provenance`, not a retuning opportunity.

## Synthetic one-hot activation

Construct exactly one BF16 activation row `x` of shape `[1,512]`, initialized to positive zero, with only `x[column] = 2^k`. Choose `k` deterministically as the smallest integer in the closed interval `[-8,8]` for which all four independently FP32-accumulated then BF16-cast scalars are finite and the original/mutated results differ:

`y = BF16(sum_j FP32(x_j) * FP32(w_row,j))`

`y' = BF16(sum_j FP32(x_j) * FP32(w'_row,j))`.

Here `w` is the decoded original Q5 matrix and `w'` differs only at the selected packed field. Because `x` is one-hot, the sum has exactly one nonzero product; the runner and independent verifier must additionally prove `y == BF16(2^k * w[row,column])` and `y' == BF16(2^k * w'[row,column])`. If no `k` in the frozen interval works, the outcome is `control_design_negative`; no wider range or new field may be tried under C1.

The selected `k`, activation SHA-256, nonzero count/index/value, original and mutated FP32 product bit patterns, BF16 output words and XOR must be retained. Require exactly one nonzero activation word, finite values, and `original_bf16_word != mutated_bf16_word`.

## Real metadata/digest rejection and unsafe bypass

Use the same semantic checker contract as S0-R5 with requested and presented metadata both fixed to expert/shared ID `512`, projection `2`, shape `[2048,512]`. The requested digest is the original codes+scales digest; the presented digest is recomputed from the actually mutated packed codes plus unchanged scales. The mutation must alter exactly one five-bit field by exactly one q step, change no scale or other field, remain in `[0,30]`, and change the combined digest.

The safe path must reject before decode/linear execution with exactly `['codes_scales_digest']`. The unsafe bypass must decode the actual mutated bytes and produce the independently predicted different BF16 scalar. A fake rejection, summary-only mutation, fabricated digest, or directly edited decoded tensor is invalid.

## Hard adjudication

`control_sensitivity_positive` requires every condition below:

1. all provenance hashes and the S0-R5 committed bundle are exact;
2. source reread, requantization, packing/unpacking and ordinal-758 evidence are exact;
3. deterministic selection reproduces `(0,0), q=6 -> 5`;
4. activation construction and smallest-`k` proof are exact;
5. original execution equals the closed-form BF16 oracle bitwise;
6. unsafe mutated execution equals its closed-form BF16 oracle bitwise and differs from original by at least one BF16 word;
7. safe metadata/digest checking rejects exactly for `codes_scales_digest` before execution;
8. every retained raw value and scalar is finite; raw schema/manifests/hashes are independently reconstructed;
9. one-thread deterministic CPU runtime, fixed dependency-lock affinity, highest matmul precision, MKLDNN enabled, denormal flushing disabled with a nonzero subnormal witness, no autocast, inference mode true and CUDA uninitialized;
10. create-new atomic result/raw/commit or failure evidence passes the already repaired Windows writable-handle fsync discipline.

Any arithmetic or provenance failure is a named negative/invalid outcome. A positive C1 repairs only the control-sensitivity gate and may be combined with immutable S0-R5 evidence by a separately preregistered adjudicator. It does not alter S0-R5 bytes/status, prove held-out generalization, validate full layer logits, or authorize performance claims.

Expected incremental resources: one real shared-down source tensor plus decoded and mutated copies, under 64 MiB working RAM; retained evidence under 1 MiB; no GPU; no model construction; no persistent Q5 bank; no network/download. The inherited hard gates remain deliberately conservative: available RAM at start at least 16 GiB, at least 2 GiB throughout, and total process peak working set at most 12 GiB. The result must state both the expected incremental envelope and these distinct hard process-level gates.
