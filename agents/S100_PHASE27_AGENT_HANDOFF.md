# S100 Phase27 agent handoff

Phase24 remains the exact active H4 parent. Phase26 overlap was real but missed
its >=2% preregistered screen gate and therefore is not silently promoted.

Phase27 targets the Phase24 grouped H-SCALE `down_gather` critical path.

Do not reopen:
- general H8 grouping;
- native BF16 substitution;
- attention/router/shared M4;
- generic cache capacity;
- Arc routed-down offload.

## Arithmetic invariant

Parent down partial for route r, chunk c:

    acc = 0
    for pi = c; pi < route_pcount[r]; pi += 8:
        p = route_plist[r, pi]
        for active column bit in ascending order:
            acc = fmaf(weight, activation, acc)

Phase27 range-down MUST execute exactly that body for the route/chunk assigned
to the current group range.

Range partitioning changes only WHEN a route/chunk is executed, never its
internal arithmetic.

After all ranges:
- use the existing `reduce_routes`;
- use the existing `accumulate_h4`;
- preserve shared term before slot0..slot5 FMAs.

## Pipeline

After `scan_group_masks`:
- main records mask-ready;
- gather stream waits mask-ready;
- gather range 0 -> ready0;
- gather range 1 -> ready1;
- ...
- main waits ready0 -> down range0;
- while main computes range0, gather stream proceeds with range1;
- ...
- main reduces all original partials once.

Each group owns a disjoint region in the existing full mirror buffer, so no
double-buffer allocation is required.

## Search ladder

1. geometry y={4,8,16,32}, one range;
2. using the best y, batches={1,2,3,4};
3. best pipeline alone vs pipeline+Phase26 shared overlap;
4. selected candidate full-state parity;
5. balanced fresh screen;
6. unchanged 5% thermal adoption.

If pipelining is insufficient, next structural route is fused zero-copy /
sparse transfer-engine work, not more group-count tuning.
