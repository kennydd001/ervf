# V12 preregistration — asynchronous harvest / queue-resident decode

Frozen after PRO-MAX V2 result intake and before any V12 target-hardware run.

Base commit: `e1dcea85bf26fe81bdf8be032a9291b8883ab659`.
Arithmetic baseline: verified V6 stack, `21.0923 ms/token = 47.4107 tok/s`.

## Observation that opens V12

PV2-20 physically measured exact K-token execution with a low-level parent graph and an individually queued control. The parent graph did not speed up execution, but the control itself was unexpectedly informative:

- K=2 individually queued child replays: p50 `18.7580 ms/token`, exact ids;
- K=4 individually queued child replays: p50 `19.0660 ms/token`, exact ids.

Both are below the 20 ms/token E50 threshold. V6's normal runner, by contrast, calls `ring_harvest()` after every token; `ring_harvest()` synchronizes the graph stream. Because the captured token graph already writes argmax to `_tok_dev`, replay N+1 does not need the host to read token N.

V12 tests whether the remaining E50 gap is primarily queue starvation caused by a host synchronization that is not causally required for decode.

## No arithmetic changes

V12 uses the exact V6 graph stack:

- device-resident routing and LRU;
- graph-safe token execution;
- selective dense ERVF;
- batched panel scan / partial reduction / exact route-order accumulation;
- batched routed up-projection ERVF;
- budget-neutral per-layer cache capacities.

No weight, kernel arithmetic, routing, precision, attention rule or model state update is changed.

## Three metrics — never conflate them

### SYNC

Current benchmark semantics: enqueue one graph replay, enqueue its D2H ring copy, call `ring_harvest()` and block until that token is on the host. This is blocking host round-trip token latency.

### QUEUED-K

Enqueue K exact autoregressive graph replays, each with the existing D2H ring copy, then synchronize once and harvest K tokens. This is **single-sequence exact generation throughput with host delivery in K-token chunks**. It is not per-token interactive host latency.

### EVENT-STREAM-K

Enqueue K graph replays without blocking. After every existing D2H ring copy, record a preallocated non-timing CUDA event on the same stream. Immediately queue the next token. The CPU polls event completion in order and reads that token's pinned ring slot only after its own event is complete.

This retains a full GPU queue while making each generated token individually visible to the host. It is the decisive arm for a strong streaming E50 claim.

## Thermal/measurement control

The PRO-MAX V2 full campaign showed monotonic BASE_A→BASE_B drift of 1.86–3.24 ms across long sequential arms. V12 therefore:

1. preheats the exact V6 graph before timing;
2. uses one captured runtime rather than recompiling between arms;
3. resets model state plus device-LRU metadata between arms without reallocating graph-bound pointers;
4. runs SYNC_A before and SYNC_B after candidates;
5. refuses a full performance conclusion if SYNC_A/B p50 drift exceeds 1.0 ms.

## Frozen queue sizes

Smoke: K = 2,4,8.

Full: K = 2,4,8,16,32 for queued throughput; K = 4,8,16 for event-stream delivery.

No K is selected after looking at results.

## Correctness gates

- SYNC_A ids == SYNC_B ids for all prompts;
- every QUEUED-K id sequence == SYNC_A exactly;
- every measured EVENT-STREAM-K id sequence == SYNC_A exactly;
- prompt staging keeps the V3 prompt-only synchronization repair;
- no model/runtime reconfiguration between arm types except reset of dynamic state.

Any token mismatch closes the corresponding arm regardless of speed.

## Performance gates

Full mode only:

- baseline drift: `abs(SYNC_A.p50 - SYNC_B.p50) <= 1.0 ms`;
- enough work: at least 500 exact queued decode tokens across the full campaign;
- **Queued E50 candidate:** exact QUEUED-K aggregate wall-clock throughput >=50.0 tok/s on one sequence;
- **Streamed E50 candidate:** exact EVENT-STREAM-K throughput >=50.0 tok/s AND the median across prompts of each prompt's p50 host-delivery gap <=20.0 ms.

The latter is the first result in this project that may be described as a streaming single-sequence E50 candidate. It still requires long-run/thermal validation before adoption.

## What does not count

- The child-parent graph's ~51 tok/s number is not reused as a V12 result.
- No component estimate is added to V6.
- Queued-K alone is not called interactive latency.
- Event completion polling may cost CPU; that cost is included in EVENT-STREAM wall time.
- A capability/API success does not upgrade a throughput result.

## Follow-up if EVENT-STREAM loses the queued gain

If QUEUED-K >=50 but EVENT-STREAM-K <50, the next exact scheduler experiment is a device-published mapped pinned ring with a monotonically increasing sequence word, eliminating per-token CUDA event records/queries while preserving safe host visibility. That experiment is not silently substituted into V12.
