# PRO V13 — steady/interleaved exact mixed Q/K/V remeasurement

Frozen before any V13 target-hardware run.

PV2-11 already established:

- Q/K/V direct outputs bitexact;
- full causal token parity on all three prompts;
- candidate kernel captured structurally;
- previous full p50s `20.7964 / 21.4866 / 22.6541 ms` for BASE_A / QKV /
  BASE_B, giving a candidate 0.23865 ms below the baseline midpoint;
- BASE_A/B drift was 1.8577 ms, so that speed attribution was correctly
  rejected as unstable.

V13 does not change the candidate arithmetic. It changes only the measurement
harness to decide whether the small signal is real.

## Candidate

One launch has three block regions:

1. Q rows use the same width-16 exact-reduction virtual-fusion mapping already
   verified in PV2-11/V6 selective ERVF.
2. K rows use the production 256-thread BF16 GEMV reduction.
3. V rows use the same production reduction.

Only launch aggregation changes.

## Fixed block schedule

Eight measured blocks, treatment order:

`BASE, QKV, QKV, BASE, QKV, BASE, BASE, QKV`

The mean block position is identical for both treatments (4.5). Every block:

1. discards the previous graph;
2. rebuilds the same non-uniform V6 cache state;
3. captures the requested treatment;
4. runs a fixed preheat on the first anchor prompt;
5. measures the same three prompts.

No treatment order is chosen after seeing timings.

Smoke: 3 prompts x 32 tokens/block, preheat 48 tokens.
Full: 3 prompts x 128 tokens/block, preheat 96 tokens. Four blocks per treatment
produce >1500 timed decode samples per treatment.

## Correctness gates

- direct Q, K and V outputs bitexact;
- all four BASE block token sequences identical to BASE block 1;
- all four QKV block token sequences identical to BASE block 1;
- candidate graph DOT contains `v13_qkv_mixed_fused`;
- no technical failure.

## Stability gate

For full-mode performance attribution:

- range of the four BASE block p50 values <=1.0 ms;
- range of the four QKV block p50 values <=1.0 ms.

If either range exceeds 1.0 ms, performance remains `unresolved_unstable` even
if the pooled candidate looks faster.

## Speed gates

Let `B` be the median of four BASE block p50 values and `C` the median of four
QKV block p50 values.

- no material regression: `C <= B * 1.002`;
- positive candidate: `B-C >= max(0.10 ms, 0.5% * B)`.

The candidate can be recommended for composition only if correctness, stability
and positive-candidate gates all pass. The gain is never added arithmetically to
V12 scheduler gains; a later combined physical run is required.
