# PRO V12C — rolling credit window with per-token event wait

Frozen before target-hardware execution.

V12B busy-queries the oldest completion event. V12C changes exactly one host
scheduler primitive: after initially filling a credit window, the CPU calls
`synchronize()` on the **oldest token's CUDA event**, not on the CUDA stream.
The event is ordered after that token's D2H ring copy. Later graph replays are
already queued after the event on the same stream, so waiting for the event does
not require later work to finish and does not intentionally drain the stream.

This tests a production-relevant alternative to busy polling that can reduce CPU
package power and scheduler noise on a laptop while preserving per-token host
delivery.

## Arms

- `SYNC_A`: existing whole-stream blocking harvest after every token.
- rolling event-wait windows `W={1,2,4,8,16,32}`.
- `SYNC_B`.

V6 arithmetic, prompt-staging repair, graph, weights, routing, cache policy and
argmax are unchanged.

## Correctness/safety gates

- SYNC A/B token ids identical for every prompt.
- every event-wait arm token id identical to SYNC_A.
- event object reused only after its preceding synchronize returned.
- max outstanding <= W < ring size.
- full-mode SYNC A/B p50 drift <=1.0 ms.

## E50 streamed gate

In full mode, a window verifies `E50_event_wait` only if:

- >=500 decode tokens across prompts;
- all ids exact;
- aggregate host-delivered throughput >=50.0 tok/s;
- every prompt's steady-state p50 delivery gap <=20.0 ms;
- baseline drift <=1.0 ms.

Steady-state gap trimming is identical to V12B:
`min(W, floor(number_of_gaps/4))` initial gaps discarded.

First-delivery delay and the time spent inside event synchronize are reported,
not hidden.

## Claim boundary

A pass is exact single-sequence greedy **streaming throughput** with individually
host-visible tokens while future causal replays may already be queued. It is not
a claim that arbitrary host logic can run between tokens without reducing that
throughput, and it is not a 75/100 tok/s claim.
