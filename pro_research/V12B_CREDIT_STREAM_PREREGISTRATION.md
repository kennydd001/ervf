# PRO V12B — rolling credit-window exact streaming

Frozen: 2026-08-16, before any target-hardware V12B result.
Base branch: `pro-v12-async` after V12 construction.
Target: `models/nemotron_3_5_lightning_v35`.

## Motivation

PV2-20 showed that the unchanged exact V6 token graph reaches roughly 18.8–19.1
ms/token when multiple causal replays are queued before a blocking host harvest,
whereas the production benchmark synchronizes the whole graph stream after every
token and measures about 21.1 ms/token. V12 separates synchronous, fixed-K queued,
and per-token event delivery.

V12B tests one additional scheduler mechanism: **never intentionally drain the
GPU queue between epochs**. A fixed credit window W is initially filled. The CPU
polls the completion event associated with the oldest ring slot; as soon as that
single token's D2H copy is complete, the token is delivered and exactly one new
causal graph replay is appended at the tail using the now-free event object.
This keeps at most W exact token replays outstanding until the rollout ends.

No model arithmetic, weight placement, routing, cache policy, quantization,
argmax or graph body is changed.

## Causal invariant

The captured graph is already a one-token recurrence:

`tok_dev[n] -> graph -> argmax -> tok_dev[n+1]`.

All graph replays, D2H token copies and event records are ordered on the same
CUDA stream. Therefore host delivery of token n is not required to choose token
n+1. Host-side credit-window scheduling changes only when later graph replays are
submitted, not any data dependency inside the recurrence.

Every V12B output id must equal the blocking SYNC reference id at the same
position. Any first divergence closes the candidate regardless of speed.

## Arms

One V6 runtime/graph is constructed and preheated before timing.

- `SYNC_A`: blocking `ring_harvest` after every token.
- `CREDIT_W1`: event delivery with exactly one outstanding replay. This isolates
  `stream.synchronize()` versus event-query overhead without queue-depth benefit.
- `CREDIT_W2`, `W4`, `W8`, `W16`, `W32`: rolling credit windows.
- `SYNC_B`: same blocking reference after all candidate arms.

Prompt staging retains the V3 prompt-only synchronization fix and is excluded
from decode timing.

## Ring/event safety

- `W <= 32`, while the runtime ring has 8192 slots.
- The runner refuses `W >= ring_size`.
- A ring slot is read only after its event, recorded after that slot's D2H copy,
  reports complete.
- An event object is reused only after its previous record has completed and the
  corresponding token has been consumed.
- Maximum observed outstanding depth is recorded and must never exceed W.

## Measurements

Per prompt/window record:

- exact ids and first divergence;
- total generated-token throughput;
- time from first decode enqueue to first host-delivered token;
- raw host-delivery timestamps/gaps;
- all-gap p50/p95/p99;
- steady-state gap p50/p95/p99;
- number of event-query polls;
- cumulative CPU enqueue time;
- maximum outstanding depth.

Steady-state gaps discard exactly
`min(W, floor(number_of_gaps / 4))` initial gaps. This rule is frozen here and
cannot be tuned after seeing results.

## Full-mode gates

Correctness:

1. `SYNC_A ids == SYNC_B ids` for every prompt.
2. Every credit-window id sequence equals `SYNC_A` for every prompt.
3. No ring/event safety violation.
4. At least 500 credit-window decode tokens are measured in the winning arm.

Measurement stability:

5. `abs(SYNC_A.p50 - SYNC_B.p50) <= 1.0 ms`.

Separate claims:

- `E50_queued_device_generation` is *not* adjudicated by V12B; V12/PV2-20 cover
  batched-harvest throughput.
- `E50_streamed_credit` requires, for one W:
  - exact ids on every prompt;
  - aggregate host-delivered throughput >= 50.0 tok/s;
  - **each prompt** has steady-state p50 delivery gap <= 20.0 ms;
  - measurement-stability gate passes.
- `E50_sync` requires the midpoint of `SYNC_A/SYNC_B` p50 <= 20.0 ms.

First-token host latency is reported but deliberately not folded into the
steady-state E50 streaming gate. It remains a separate latency metric.

## Claim boundary

Passing `E50_streamed_credit` supports: "one exact greedy autoregressive
sequence streams >=50 generated tokens/s after warmup while every token is
individually made available to the host."

It does **not** claim <=20 ms blocking request/response latency for an
application that must run arbitrary host logic between every generated token.
It does not imply 75 or 100 tok/s. Those require new physical measurements.
