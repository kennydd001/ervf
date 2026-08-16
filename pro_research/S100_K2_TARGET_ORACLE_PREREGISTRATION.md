# S100-K2 target-verification oracle preregistration

Date: 2026-08-16
Branch: `pro-s100-mtp`
Frozen before any K2 target-block measurement.

## Question

Can the exact Lightning target consume/verify two already-known correct consecutive tokens in a single layer-major block cheaply enough that depth-1 native MTP still has a physical path to 100 tok/s?

This experiment deliberately uses **oracle draft tokens** taken from the baseline greedy sequence. It does not measure MTP prediction quality or acceptance. That isolates the target verifier, which is the necessary condition for any exact speculative speedup.

## Why K=2

The exact checkpoint has `num_nextn_predict_layers=1`. NVIDIA AutoModel defines this as one MTP forward depth and returns one hidden state per depth. The local depth has two physical inner sublayers (`attention`, `moe`). One native depth therefore does not justify assuming K=4/K=8 proposals.

K2 is the relevant first verification block: one guaranteed/known next-token position plus one MTP-proposed future position, with the target deciding the proposal.

## Frozen external anchors

- Kimi MTP inventory: `pro-e100-batch@02e5980`, 270 tensors / 2.487 GiB, official NemotronV3 name alignment, one logical depth.
- Kimi K1 state proof: `pro-e100-batch@458725d`, prefixes 0..4 exact, sabotage diverged.
- Current single-stream record: `pro-research` V18, candidate p50 **19.6046 ms/token** (51.0084 tok/s), exact token parity over 765 timed samples.
- Post-Kimi bound: `pro_research/S100_MTP_POST_KIMI_BOUND_2026-08-16.md`.

## Oracle construction rules

1. Generate the reference greedy sequence with the current target runtime and freeze the next two token ids before candidate timing.
2. Candidate starts from an identical target state snapshot at the same logical position.
3. Candidate consumes the two frozen correct input tokens in one **layer-major K=2 target block**. It may reorganize work across the two positions, but may not use MTP output to skip target computation.
4. Hybrid causality must be preserved:
   - Mamba recurrence position 0 -> position 1;
   - attention position 1 may attend position 0 and all prefix KV, never future data;
   - MoE routing is per position using that position's hidden state;
   - residual/norm order remains the model's order.
5. Candidate must produce the target logits/tokens needed to decide the K=2 block and the state needed to continue generation.
6. No draft runtime, no acceptance heuristic, no arithmetic speedup multiplication.

## Arithmetic / equivalence policy

Two candidate classes are allowed but must be labeled separately:

### A. `BITEXACT_K2`
Every persisted target state and every target logit/output checked by the harness is bit-identical to two sequential baseline steps.

### B. `TOKEN_EQUIV_K2`
A mathematically equivalent block implementation may use a different reduction order (e.g. Tensor Core GEMM). It is **not** called bitexact. It may proceed only if:

- greedy token ids equal sequential baseline on every held-out block;
- deterministic repeat gives the same candidate ids;
- no NaN/Inf;
- final logical position and state shapes are valid;
- a separate sabotage/control changes at least one token;
- full validation is at least 10,000 target tokens before any adoption claim.

The first smoke/full oracle may establish only a `TOKEN_EQUIV_K2_candidate`; not adoption.

## Measurement arms

Same process / same loaded target / matched thermal era:

- `SEQ_A`: two ordinary sequential target steps per block.
- `K2`: one layer-major two-position verifier block.
- `SEQ_B`: sequential repeat.

Use preheat before `SEQ_A`. Full mode must interleave short blocks or otherwise keep `SEQ_A/SEQ_B` drift <= 1.0 ms **per two-token block**. Do not loosen this threshold post-hoc.

Report both:

- ms per two-token block;
- effective verified tok/s = `2000 / block_ms`;
- per-position delivery semantics separately; this is verification throughput, not yet streamed user tok/s.

## Correctness gates

- `G1_reference_A_B_token_parity` — SEQ_A and SEQ_B identical.
- `G2_candidate_token_parity` — K2 ids identical to SEQ_A for every block.
- `G3_deterministic` — repeated K2 identical.
- `G4_state_valid` — target state can continue for >=32 ordinary steps and those continuation ids match baseline.
- `G5_control_diverges` — a deliberately wrong second oracle token or corrupted state produces detectable divergence.
- `G6_no_nan_inf`.

If any G1-G6 fails, performance is not interpreted.

## Performance gates

The 100-tok/s iteration budget with at most two useful tokens is 20.000 ms.

From the post-Kimi byte bound, even a fully resident MTP active-byte floor is ~0.715 ms; therefore:

- `P1_K2_block_lt_19_285ms`: necessary for S100 under the optimistic resident-draft floor.
- `P2_K2_block_lt_17_500ms`: strong candidate, leaves >=2.5 ms for real draft/acceptance overhead.
- `P3_effective_verified_ge_110tps`: report-only strong marker (`block_ms <= 18.182 ms`).
- `P4_speedup_vs_seq_mid_ge_1_50x`: architecture-value gate; K2 must save substantial target work, not merely move noise.

`P1` is the minimum feasibility gate. Passing P1 does **not** claim S100; it only permits implementing the real depth-1 MTP drafter. Failing P1 closes native depth-1 MTP as an S100 mechanism on the current target stack unless the target itself is first accelerated.

## Stop rules

- If a bitexact implementation is >= sequential cost, do not endlessly tune it under this preregistration.
- If token-equivalent Tensor Core/block kernels are needed, freeze their own numerical-validation preregistration before target data.
- Do not build the real MTP drafter until K2 clears P1 with all correctness gates.
- Do not merge to `main` or `pro-research` from this branch.
