# S100 Phase41 — full H8 group-range MoE pipeline

Frozen before Phase41 GPU timing.

## Opens from

Phase40's exact three-range gather/down pipeline improved the Phase32 H8 parent
by 0.95%, from a 123.384 ms baseline midpoint to 122.210 ms. It failed the 3%
promotion gate because all routed-UP work and the global mask scan still finish
before any gather starts. Phase25 profiling priced routed-UP at about 28.1 ms/H8
and gather at about 22.8 ms/H8.

## Frozen candidate

`FULL_PIPELINE_B3` keeps Phase32 `dense_m8` and the fixed group ranges
`(0,16)`, `(16,32)`, `(32,48)`, but streams every range through:

1. exact existing grouped routed-UP kernels for multiplicities 1 through 8;
2. exact existing group mask/union scan;
3. Phase40's exact range gather (`grid.y=4`);
4. Phase40's exact range routed sparse-down.

UP/scan execute sequentially on an UP stream. Gather executes on a second
stream after that range's scan event. Sparse-down executes on a third stream
after that range's gather event. The final existing route reduction and exact
slot-0-to-slot-5 accumulation wait for all three down events.

The existing UP and scan kernels are range-launched by slicing only their
`group_count`, `group_refs`, and union-output pointers. Route references remain
global. Kernel arithmetic, block geometry, cache state, route order, partial
indices, reduction order and accumulation order are unchanged.

## Protocol and gates

Fresh-process `BASE_A`, `FULL_PIPELINE_B3`, `BASE_B`; canonical context 1024;
four warmup plus sixteen measured H8 windows.

- `G41-C1`: all arms produce every canonical token exactly.
- `G41-R1`: Phase40's only new CUDA kernels retain zero local-memory bytes.
- `G41-D1`: baseline drift <= 5% of baseline midpoint.
- `G41-P1`: candidate median improves by at least 3%.
- `G41-P2`: candidate <= 120 ms/H8 (strong milestone, reported separately).

Passing C1/R1/D1/P1 opens state and four-round thermal promotion. This remains
target-only verifier timing; S100 requires <=80 ms/H8 before drafter cost.

