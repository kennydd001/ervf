# S100-KVERIFY preregistration — exact multi-token target-verifier oracle

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: no S100-KVERIFY target result exists.

## Purpose

Single-stream E100 cannot be inferred from aggregate batching. Native MTP is useful only if the **target model can verify multiple proposed positions fast enough** while preserving the baseline greedy sequence and persistent state.

KVERIFY therefore separates verifier physics from draft quality. Its full oracle receives K tokens that are known in advance to be the baseline-correct continuation. There are no rejections and no MTP acceptance-rate advantage. The question is only whether the target can process/verify those K positions layer-major faster than K independent token steps.

A passing oracle is not an MTP speed claim. A failing oracle closes the MTP performance branch on this runtime until the target verifier changes.

## Connection to E100-MRHS

Layer-major K-token verification reuses the same common matrix across K token positions. Exact MRHS is therefore directly applicable to Q/O, Mamba in/out, routers, shared experts and LM head. Routed token/expert pairs can use PAIRBATCH and later an explicitly preregistered expert-major MRHS arm when the same expert occurs at multiple positions.

No MRHS component speedup is projected into KVERIFY. Only integrated verifier measurement counts.

## Phase K0 — state/rollback budget

GPU-free metadata computes:

- mutable Mamba SSM-state bytes;
- mutable Mamba conv-state bytes;
- one full Mamba checkpoint bytes;
- per-token/per-layer Mamba in-projection output bytes;
- K={2,4,8} bytes for one initial checkpoint plus stored in-projection outputs;
- speculative FP8 KV append bytes (diagnostic; KV can roll back by position).

No pass/fail speed claim is made at K0.

## Phase K1 — exact Mamba rollback proof

For a warmed real Mamba state and K=4 real consecutive token activations:

1. save one initial copy of `ssm[layer]` and `conv[layer]`;
2. run the normal Mamba path for K positions and record the in-projection output `proj` for each position;
3. for every accepted prefix j in `{0,1,2,3,4}`, restore only the initial state and replay **state transition only** for the first j stored `proj` values:
   - split stored `proj` into z/xbc/dtr;
   - rerun the exact production `conv_step` on xbc;
   - rerun exact `dt_activate` on dtr;
   - rerun exact `ssm_step` on x/B/C/dt;
   - `gated_norm`, output projection, residual, MoE, attention and LM head are deliberately skipped because they are not needed to reconstruct persistent Mamba state after already-verified accepted inputs.
4. compare reconstructed `conv` and `ssm` bitwise against reference state snapshots captured after j normal transitions.

All j must be bit-identical. A sabotage arm skips one accepted transition and must differ for j>0.

This proves the rollback/commit mechanism only; it does not measure full verification.

## Why one state snapshot is enough if K1 passes

The normal Mamba transition is determined by prior `conv`/`ssm` state plus the current layer's `in_proj` output and fixed layer parameters. Therefore a speculative block may retain one initial state snapshot and K stored `in_proj` outputs rather than K full state snapshots. On partial acceptance, restore the initial state and replay only accepted transitions. This is a hypothesis until K1 bitwise proof passes.

## Phase K2 — full target-verifier correctness

Implement a fixed K=4 teacher-forced verifier over the complete 52-layer target. Input tokens are a previously recorded baseline continuation, not MTP proposals.

Mandatory correctness:

- hidden/logit argmax at every verified position equals K sequential adopted-baseline steps;
- final KV position/state equals sequential execution after K positions;
- final Mamba state is bit-identical after all-K accept;
- K1 rollback path reconstructs every partial accepted prefix state exactly;
- deterministic repeat exact;
- sabotage one provided token and observe downstream verification/token mismatch.

No speed interpretation if any gate fails.

## Phase K3 — verifier throughput oracle

Full campaign: K in `{2,4,8}`, >=1000 total verified positions per K, A/B/A or balanced interleaving against K sequential adopted V6 steps, raw timing retained, baseline drift <=1 ms/token-equivalent or <=5% batch time (use the stricter condition when both apply).

Define:

`verified_positions_per_second = 1000 * K / verifier_batch_ms`.

A **single-stream E100 verifier oracle** requires:

- exact K2 gates;
- at least one preregistered K reaches >=100 verified positions/s;
- its p95 batch time is reported;
- extra VRAM is measured and <= the actual free headroom gate established immediately before the run;
- an independent verifier recomputes throughput and correctness status.

This says only that a perfect draft could be verified at E100. It is not realized generation throughput.

## Phase K4 — MTP only after K3

Only if local Lightning MTP name/semantics mapping passes and K3 reaches E100 may MTP drafting be attached. Realized exact throughput must then include draft cost, target verification, rejection/commit cost and actual acceptance distribution. No acceptance statistic from another Nemotron checkpoint is transferable to Lightning.
