# S100-K2 V18 target-verification oracle preregistration

Date: 2026-08-16
Branch: `pro-s100-k2-oracle-v18`
Base: `pro-research@43720efbb202c115b49413e13157dad4867093bf`; adopted performance reference remains V18 `2bd156bcee1db35103d314c0bede36cc51fc6c8a`.
Frozen before any K2 measurement on this branch.

## Question

Can the exact V18 Lightning target consume/verify two already-known correct consecutive tokens in one **layer-major K=2 block** cheaply enough that depth-1 native MTP retains a physical path to 100 tok/s?

The proposed tokens are oracle tokens frozen from the target's own greedy sequence. This experiment therefore measures only target verification; it makes no MTP acceptance or draft-quality claim.

## Why this branch exists

The older `pro-s100-mtp` preregistration was created on the diverged `pro-e100-batch` lineage. The current record is V18: 19.6046 ms/token = 51.0084 tok/s, exact token parity. K2 must be judged on the current target stack, including selective ERVF, the adopted batched MoE path, H-SCALE and B3 overlap. No stale pre-V18 performance baseline may hard-close S100.

## Frozen anchors

- V18 record commit `2bd156bcee1db35103d314c0bede36cc51fc6c8a`: 19.6046 ms/token in run 2, 19.6897 ms/token in run 1, exact over 3 prompts x 765 timed tokens per arm.
- Kimi MTP inventory `pro-e100-batch@02e5980`: one logical MTP depth, 270 tensors / 2.487 GiB.
- K1 rollback proof `pro-e100-batch@458725d`: Mamba conv+SSM rollback exact for prefixes 0..4; sabotage diverged.
- Analytic resident MTP active-byte floor: ~0.715 ms/draft. This is a lower bound, not measured runtime.
- Native-NVFP4 C0B `pro-s100-nativefp4@92dc8eb`: all 5,935 audited NVFP4 weight/scale pairs are logically group-16 once packed 2-codes/byte storage is counted correctly. Native Tensor Core execution remains a separate branch and is not used here.

## K2 construction

For each prompt, first freeze a greedy target sequence. Candidate input pair for block b is `[t[2b], t[2b+1]]`; expected target outputs are `[t[2b+1], t[2b+2]]`.

The K2 candidate starts from an identical reset + prompt prefill and executes layers in **layer-major** order:

1. embed both known input tokens;
2. for each target layer, normalize both position states;
3. Mamba: execute position 0 then position 1 against the same persistent layer state, preserving recurrence order;
4. attention: position 0 writes/reads KV at p, then position 1 writes/reads KV at p+1; position 1 may see position 0 and the prefix, never future data;
5. MoE: execute position 0 then position 1 for that layer; routing remains natural and per-position; per-layer cache call order therefore matches sequential decoding;
6. residual/add order is unchanged per position;
7. final norm + target head + argmax are produced for both positions;
8. persistent model state after the block must equal two sequential target steps.

The first implementation is a correctness-first layer-major oracle using the already-adopted V18 kernels. It does **not** get credit for hypothetical MRHS/native-FP4 speedups. If it is too slow, the result is `layer_major_v18_negative`, not a universal proof that every future K2 kernel is impossible. A materially new common-weight/Tensor-Core K2 implementation requires its own preregistration.

## Arithmetic classes

### `BITEXACT_K2`

Required for the primary arm. Per-position kernels, FP32 recurrence, route order, fmaf/reduction order and target head are unchanged. The harness must establish:

- every generated token equals sequential V18;
- final Mamba conv and SSM states are bit-identical;
- used KV prefix bytes are bit-identical;
- final checked target logits are bit-identical;
- candidate position equals sequential position;
- deterministic replay is identical.

### `TOKEN_EQUIV_K2`

Not opened by this runner. Any later Tensor-Core/MRHS implementation that changes reduction order or activation precision must get a separate preregistration and quality/token-equivalence gates before timing.

## Measurement

Same loaded target and matched thermal era:

- `SEQ_A`: two ordinary V18 graph replays per measured block, one block harvest/sync.
- `K2`: one captured layer-major K2 graph, oracle input pair staged before replay, one block harvest/sync.
- `SEQ_B`: repeat sequential arm after candidate.

Smoke: 4 K2 blocks per prompt. Full: 64 K2 blocks per prompt. Full uses all available preregistered prompts. Timing is wall time per two-token block and includes input staging/output harvest for K2.

Report:

- raw block timings;
- p50/p95/p99;
- `effective_verified_tok_s = 2000 / p50_block_ms`;
- sequential midpoint and speedup;
- BASE_A/BASE_B drift.

## Correctness gates

- `G1_reference_A_B_token_parity`: SEQ_A == SEQ_B at every produced token.
- `G2_candidate_token_parity`: K2 == SEQ_A at every produced token.
- `G3_deterministic`: a repeated untimed K2 correctness run gives identical ids.
- `G4_state_bitexact`: final conv, SSM, used KV and checked final logits have zero bit mismatches; positions agree.
- `G5_continuation_32`: after K2, copy the candidate logical position into the ordinary V18 graph position and continue >=32 ordinary exact target steps; ids equal frozen target continuation.
- `G6_control_diverges`: a deliberately wrong second oracle input or corrupted state changes token output and/or persistent state.
- `G7_no_nan_inf`: checked K2 logits/states contain no NaN/Inf.

If any correctness gate fails, performance is not interpreted.

## Performance gates

One depth can contribute at most about two useful target positions per successful iteration. S100 therefore has a 20.000 ms total iteration budget.

- `P1_K2_block_lt_19_285ms`: necessary feasibility gate under the optimistic ~0.715 ms resident-draft floor.
- `P2_K2_block_lt_17_500ms`: strong gate, leaving >=2.5 ms for real draft/acceptance overhead.
- `P3_effective_verified_ge_110tps`: report-only marker, K2 p50 <=18.182 ms.
- `P4_speedup_vs_seq_mid_ge_1_50x`: architecture-value gate.
- `D1_seq_A_B_drift_le_1ms`: p50 drift <=1.0 ms per two-token block.

Passing P1 only authorizes building/measuring the real depth-1 drafter. It is not an S100 claim.

## Stop / claim rules

- Do not loosen G/P/D thresholds after seeing results.
- Do not multiply K2 speed by MTP acceptance estimates.
- Do not call oracle verified tok/s user-visible generation tok/s.
- Do not merge this experimental branch into `pro-research` automatically.
- If correctness fails, fix only a demonstrable implementation bug; preserve the failed artifact.
- If the correctness-first V18 layer-major candidate is >= sequential cost, close that implementation honestly. A different MRHS/native-Tensor-Core architecture must open a new registered arm rather than post-hoc tuning this one.
