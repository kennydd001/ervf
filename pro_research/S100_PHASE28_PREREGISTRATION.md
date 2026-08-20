# S100 Phase28 preregistration

## Frozen source/evidence

Required:
- Phase24 active parent adopted and state-green;
- Phase27 complete, state-green, not adopted;
- Phase27R complete, not adopted;
- Phase27R median round gain >0 and <5%;
- Phase27R NEXT_ROUTE=FUSE_GATHER_DOWN_AND_ELIMINATE_MIRROR_TRAFFIC;
- correct 52-layer Lightning snapshot;
- no local edits in Phase24/23 parent CUDA sources.

## Synthetic preflight

Fixture includes M=1,2,3,4 groups and all 24 routes.

Compare raw route/chunk partial arrays against exact Phase24:
1. Phase24 gather -> Phase24 down;
2. direct_route;
3. group_chunk_v16;
4. group_allchunks_v4;
5. group_allchunks_v16.

Required:
- byte/bit equality of every partial;
- finite;
- every active route/chunk written;
- valid panel->chunk map;
- no group multiplicity outside 1..4.

A preflight technical failure is not a hypothesis NO-GO.

## Actual alignment/resource audit

On the real checkpoint report:
- down_base_ptr mod16 for every MoE layer;
- panel stride mod16;
- row-half stride mod16;
- mirror bytes removed;
- kernel register count;
- static/dynamic shared bytes;
- local-memory bytes when exposed.

`*_v16` arms are eligible only when all real pointers/strides are aligned.

## Fresh arm screen @ context1024

Order:
- Phase24 parent A;
- Phase27 control;
- direct_route;
- group_chunk_v16;
- group_allchunks_v4;
- group_allchunks_v16;
- group_allchunks_v16_overlap;
- Phase24 parent B.

Each fresh process:
- 6 H4 warmups;
- 10 measured H4 blocks;
- exact tokens.

Parent drift <=7%.

Phase27 control is diagnostic only.

Mirrorless selection:
- exact;
- technically valid;
- lowest median;
- stable Phase24 parent bracket;
- screen gain >=2%.

Tie within 1% prefers:
1. group_allchunks_v16;
2. group_allchunks_v4;
3. group_chunk_v16;
4. direct_route;
5. overlap variant only when strictly faster, because it adds another stream.

## Full-state gate @ context1024

Fresh parent and candidate:
- IDs exact;
- deterministic candidate replay;
- SSM NRMSE <=5e-5;
- conv <=1e-5;
- FP32 KV <=5e-6;
- logits <=5e-4;
- finite.

## Final screen

PARENT_A -> CANDIDATE_A -> CANDIDATE_B -> PARENT_B.

8 warmups + 12 measured H4 blocks.

Open thermal adoption iff:
- state green;
- exact;
- parent and candidate A/B drift <=7%;
- midpoint gain >=2%.

## Thermal adoption

Non-scoring parent primer, then:

R1 P -> C
R2 C -> P
R3 C -> P
R4 P -> C

Each scored process:
- context1024;
- 8 warmups;
- 16 measured H4 blocks.

Adopt iff:
- median round gain >=5%;
- median paired gain >=5%;
- >=3/4 rounds positive;
- both robust CV <=5%;
- all exact/aligned.

## Promoted contexts

If adopted:
- context128: 4 warmup + 12 measured;
- context1024: 4 warmup + 12 measured;
- context4096: 0 warmup + 12 measured.

## Gates

PHASE28_MIRRORLESS_DOWN_ADOPTED
TARGET_100_TARGET_ONLY_OPEN:
  every promoted context <=10.000 ms/useful token.
DRAFTER_SHOOTOUT_OPEN:
  every promoted context <=8.000 ms/useful token.

S100_SINGLE_ACHIEVED=false in Phase28.
