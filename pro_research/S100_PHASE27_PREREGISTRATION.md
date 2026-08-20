# S100 Phase27 preregistration

## Frozen evidence

Phase24 is the active exact H4 parent.

Phase26:
- H4 overlap state parity: exact.
- H4 screen gain: 1.9607538623748644%.
- No thermal adoption.
- H8 screen invalid from symmetric parent/candidate trace mismatch.
- NEXT_ROUTE=BUILD_DOWN_GATHER_TRANSFER_COMPUTE_PIPELINE.

## Preflight gates

Before model timing:
1. compile Phase27 CUDA;
2. synthetic grouped gather range output is byte-identical to Phase24 full
   grouped gather;
3. synthetic range-down partials are bit-identical to Phase24 full down;
4. 2-stage cross-stream graph capture/replay succeeds;
5. report concurrentKernels and asyncEngineCount.

A preflight failure is technical evidence, not a hypothesis NO-GO.

## Geometry sweep @ context 1024

Fresh process per arm:
parent A, y4, y8, y16, y32, parent B.

4 warmups + 8 measured H4 blocks.

Candidate uses one group range (no pipeline), so only gather launch geometry
changes. It must remain token-exact.

Geometry selection:
- both parent anchors measured;
- parent relative drift <=7%;
- choose lowest candidate midpoint;
- y32 is the semantic control;
- no adoption decision is made here.

## Pipeline sweep @ context 1024

Using selected gather_y:
parent A, batches=1,2,3,4, parent B.

4 warmups + 8 measured H4 blocks.

Fixed group ranges partition [0,24) as evenly as possible.

## Combination sweep

Using selected gather_y + batches:
parent, pipeline only, pipeline + exact Phase26 shared overlap, parent.

6 warmups + 10 measured H4 blocks.

Select lower ms/useful-token candidate. Do not add historical Phase26 gain.

## Full-state gate @ context 1024

Fresh parent and selected candidate from identical prefix.

Required:
- IDs exact;
- deterministic candidate replay IDs;
- max SSM NRMSE <=5e-5;
- max conv NRMSE <=1e-5;
- max FP32 KV NRMSE <=5e-6;
- logits NRMSE <=5e-4;
- finite.

## Final screen @ context 1024

PARENT_A -> CANDIDATE_A -> CANDIDATE_B -> PARENT_B
8 warmups + 12 measured H4 blocks.

Stable if each A/B arm drift <=7%.

Thermal stage opens iff:
- state green;
- all tokens exact;
- stable;
- midpoint gain >=2%.

## Thermal adoption

Non-scoring parent primer then:
R1 P -> C
R2 C -> P
R3 C -> P
R4 P -> C

8 warmups + 16 measured H4 blocks per arm.

Adopt iff:
- median round gain >=5%;
- median position-paired gain >=5%;
- >=3/4 rounds positive;
- parent robust CV <=5%;
- candidate robust CV <=5%;
- all exact.

## Promoted contexts if adopted

128:  4 warmup + 12 measured H4 blocks
1024: 4 warmup + 12 measured H4 blocks
4096: 0 warmup + 12 measured H4 blocks

## S100 gates

TARGET_100_TARGET_ONLY_OPEN:
  all promoted contexts <=10.000 ms/useful token.

DRAFTER_SHOOTOUT_OPEN:
  all promoted contexts <=8.000 ms/useful token.

S100_SINGLE_ACHIEVED=false in Phase27.
