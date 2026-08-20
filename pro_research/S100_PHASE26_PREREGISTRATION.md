# S100 Phase26 preregistration

## Parent
Phase24 selected best-of-all is frozen:
- 23/23 resident H-SCALE planes;
- attention_m4=false;
- router_m4=false;
- shared_m4=false;
- production_x4 head;
- GPU grouped H4 MoE.

Phase25 H8 remains research-only.

## Synthetic preflight
Capture/replay a graph with:
- main stream work;
- event fork;
- side stream work;
- event join;
- dependent final kernel.
Exact expected output required.

## State parity @1024

H4:
Phase24 parent vs H4 overlap from identical prefix.

H8:
Phase24 H4+H4 parent vs H8 direct8 overlap from identical prefix.

Required for each candidate:
- IDs exact;
- deterministic candidate replay IDs;
- max SSM NRMSE <=5e-5;
- max conv NRMSE <=1e-5;
- max FP32-KV NRMSE <=5e-6;
- tail logits NRMSE <=5e-4;
- finite.

## Same-era screens @1024

For H4:
PARENT_A -> OVERLAP_A -> OVERLAP_B -> PARENT_B.

For H8:
PARENT_A -> OVERLAP_A -> OVERLAP_B -> PARENT_B.

Fresh process per arm.
8 warmup windows + 12 measured windows.

A screen is stable if each arm A/B relative median drift <=7%.

Candidate screen gain:
  1 - candidate_midpoint_ms / parent_midpoint_ms.

Selection:
- correctness/state green;
- stable;
- screen gain >=2%;
- choose lowest candidate ms/useful-token.
If no candidate clears 2%, stop before thermal adoption.

## Thermal adoption

Selected horizon only.
Non-scoring parent primer, then:
R1 P -> C
R2 C -> P
R3 C -> P
R4 P -> C

16 measured windows, 8 warmup windows @1024.

Adopt iff:
- all exact;
- positions aligned;
- median round gain >=5%;
- median paired-window gain >=5%;
- >=3/4 positive rounds;
- parent robust CV <=5%;
- candidate robust CV <=5%.

## Promoted contexts

128 / 1024 / 4096.

H4:
12 measured windows; warmup 4 at 128/1024, 0 at 4096.

H8:
12 windows at 128/1024.
At context4096 only 6 windows, no warmup, because the frozen canonical trace
ends at token 4145.

## Gates

PHASE26_ACTIVE_PARENT_ADOPTED
TARGET_100_TARGET_ONLY_OPEN:
  <=10.000 ms/useful token at every promoted context.

DRAFTER_SHOOTOUT_OPEN:
  <=8.000 ms/useful token at every promoted context.

S100_SINGLE_ACHIEVED=false always in Phase26 because there is no drafter.
