# S100 Phase45 — persistent split routed-UP workerpool

## Frozen parent

`codex/s100-phase31-critical-path@1046da1`, 63.53125 ms/H4 and 62.961
target-only tok/s at context 1024.

## Mechanism

Keep Phase30E's exact M1-2 and M3-4 specialization split, but replace the
116-row-tile grid per possible expert group with a small fixed workerpool.
Each CTA stages the same activation rows once and then processes multiple
16-row tiles in ascending order.  Every output row retains the production
virtual-lane mapping, FMA sequence, reduction tree and ReLU-squared operation.

Frozen schedules:

- `p2_p4`: 2 workers/group for M1-2, 4 workers/group for M3-4;
- `p4_p4`: 4 workers/group for both kernels;
- `p4_p8`: 4 workers/group for M1-2, 8 workers/group for M3-4.

The possible CTA positions fall from 5,568 per MoE layer to respectively
144, 192 and 288.  Inactive expert groups still return without arithmetic.

## Initial screen

- context 1024;
- 4 warmup and 8 measured H4 blocks per isolated process;
- order parent A, P2/P4, P4/P4, P4/P8, parent B;
- exact token IDs required;
- zero local-memory bytes required for both persistent kernels;
- parent drift at most 5%.

Only a schedule with at least 2% interpolated integrated gain proceeds to
state capture and four thermal rounds.  Final adoption additionally requires
bit/state/logit parity, 4/4 positive rounds, median gain at least 2%, bootstrap
lower-95 gain at least 1%, and exact contexts 128/1024/4096.

## Claim boundary

Exact target-only H4 verifier.  The first screen changes routed-UP CTA
ownership only; no routing, cache, transfer, DOWN, attention, Mamba or head
arithmetic changes.
