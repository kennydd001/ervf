# S100 Phase28 handoff

## Scientific boundary

Phase24 remains the comparator. Phase27R is positive research evidence, not the
active parent.

Phase28 may remove only:
- sparse code gather kernel;
- temporary device mirror write;
- temporary device mirror read.

It must not change model arithmetic.

## Exact route/chunk invariant

For route r and chunk c, the parent computes:

```text
acc = 0
for pi = c; pi < route_pcount[r]; pi += 8:
    p = route_plist[r, pi]
    scale = E4M3(plane[slot,p,row]) * global_down_scale
    for active column bit c in ascending order:
        q = mapped_host_code[expert,p,column,row_pair]
        w = E2M1(nibble(q,row_parity)) * scale
        acc = fmaf(w, activation[r,p,column], acc)
```

Every candidate must write the same eight route/chunk partials. Existing
`reduce_routes` and `accumulate_h4` remain unchanged.

## Arm-specific invariants

### direct_route

This is the simplest exact zero-mirror control: the Phase24 down-kernel body is
copied verbatim, but `bank` points directly into the expert's mapped host
record rather than into a device mirror.

### group_chunk_v16

Grid:
- x = 128 output rows;
- y = group × 8 chunks.

For every chunk step, routes retain their own `pi=c+step*8`. When multiple
routes in a group use the same panel at that exact step, the panel's required
code columns are fetched once into shared memory and reused. No route's panel
sequence changes.

### group_allchunks

One block owns:
- one expert group;
- one 128-row tile;
- all eight chunk accumulators.

Panels are visited in ascending panel index. `panel_chunk[r,p]` maps every
route panel to its original `pi mod 8`. Since route panel lists are ascending,
the FMA sequence inside each independent chunk remains identical to the
parent, even though different chunks are interleaved in wall-clock execution.

## Alignment

The 16-byte arm is allowed only if every actual routed-down base pointer and
all relevant strides are naturally 16-byte aligned. Otherwise it is marked
infeasible and the 4-byte arm remains valid.

## Adoption

Arm screen minimum: 2% stable gain against fresh Phase24 anchors.

Production adoption remains:
- median round gain >=5%;
- median 64-position paired gain >=5%;
- >=3/4 rounds positive;
- parent/candidate robust CV <=5%;
- exact full state and logits.

Do not lower the gate.
