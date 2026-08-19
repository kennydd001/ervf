# Phase23 preregistration

Correctness @1024:
- IDs exact;
- SSM <=5e-5 NRMSE;
- conv <=1e-5;
- FP32 KV <=5e-6;
- logits <=5e-4;
- finite;
- candidate graph deterministic.
Group counts must all be 1..4.

Same-era graph screen, fresh processes:
PARENT_A -> GROUPED_A -> GROUPED_B -> PARENT_B.
Context1024, 16 blocks, 4 warmup blocks.
Parent and grouped A/B relative median drift <=7%.
Promote only if grouped midpoint <= parent midpoint * 0.95.

Promoted contexts: 128/1024/4096, 12 H4 blocks.
At 4096 no advancing warmup due canonical trace length.

PHASE23_TARGET_40MS_OPEN: all <=40 ms/H4.
DRAFTER_SHOOTOUT_OPEN: all <=32 ms/H4.
PHASE24_MOE_NEXT_OPEN: grouped >=5% faster, correct, target40 false.
S100_SINGLE_ACHIEVED=false.
