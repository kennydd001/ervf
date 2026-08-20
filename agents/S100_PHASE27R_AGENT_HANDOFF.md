# S100 Phase27R handoff

Do not modify Phase27 CUDA, launch geometry, selected variant or parent.

The selected candidate is frozen:

```text
gather_y       = 4
batches        = 3
shared_overlap = true
```

Phase27 already proved:
- IDs exact;
- full state green;
- candidate A/B stable;
- the final parent A/B bracket unstable.

The only admissible change in Phase27R is measurement order and thermal
stabilization.

## Balanced schedule

```text
non-scoring parent primer

R1 Parent    -> Candidate
R2 Candidate -> Parent
R3 Candidate -> Parent
R4 Parent    -> Candidate
```

Every scored process starts from context 1024 and advances exactly:

```text
8 warmup H4 blocks
16 measured H4 blocks
```

Thus parent and candidate in every round time identical canonical positions.

## Adoption

Adopt only when every preregistered gate is green:

- all 8 scored runs measured and token-exact;
- all four parent/candidate position lists align;
- median round gain >=5%;
- median over 64 position-paired blocks >=5%;
- at least 3/4 rounds positive;
- parent robust CV <=5%;
- candidate robust CV <=5%.

Do not lower the 5% threshold.

## Interpretation

Adopted:
- Phase27 becomes the active H4 parent;
- promoted contexts 128/1024/4096 run automatically;
- next engineering route is profiling the adopted parent, then fused
  gather/down mirror elimination.

Not adopted but positive:
- retain Phase27 as exact research evidence;
- proceed to fused gather/down mirror elimination on the Phase24 parent.

Negative:
- retain Phase24 parent;
- proceed directly to zero-copy/device-transfer down research.
