# Phase26 agent handoff

Do not re-open generic attention/router/shared M4 kernels. Phase24 closed them.

Do not assume H8 is better. Phase25 falsified general H8 as an economic win
against a fresh Phase24 parent.

Phase26 attacks the measured dominant MoE structure by concurrency.

## Exactness rule

The shared branch may execute concurrently with routed-up/gather/down because
they do not consume one another.

However the final route accumulation order MUST remain:

    out = shared_out
    for slot in 0..5:
        out = fmaf(route_down[slot], route_weight[slot], out)

Do not replace this by:

    out = shared_out + sum(routes)

because that changes floating-point association.

## CUDA graph

Phase24 already captures the cache copy stream. Phase26 adds one explicit
non-blocking shared stream per verifier and joins it with CUDA events before
route accumulation.

The candidate graph must fail closed if cross-stream capture is unsupported.

## Horizons

H4 overlap:
  Phase24 exact parent + shared/routed fork/join.

H8 overlap:
  Phase25 exact direct8_route + same fork/join.

Only same-era measurements determine adoption.

## Adoption

The selected horizon must still clear:
- median round gain >= 5%;
- median position-paired gain >= 5%;
- >=3/4 rounds positive;
- parent and candidate robust CV <=5%;
- all tokens exact;
- state gates green.

If neither horizon clears the screen, the next route is down-gather
transfer/compute pipelining, not more general route grouping.
