# Phase22 preregistration

22A lm_head:
generic_m4 vs production_x4 on real H4 final-normalized rows.
Green: finite, NRMSE <=5e-4, row argmax agreement 1.0.
Select fastest green; ties <=1% prefer production_x4.

22B graph correctness @1024:
eager repaired V6 H4 versus graph H4 from identical canonical prefill.
Required:
- four ids exact;
- max SSM NRMSE <=5e-5;
- max conv NRMSE <=1e-5;
- max FP32 KV NRMSE <=5e-6;
- H4 logits NRMSE <=5e-4;
- deterministic graph replay ids.

22C same-era:
EAGER_A -> GRAPH_A -> GRAPH_B -> EAGER_B, fresh process per arm,
8 advancing H4 blocks. Eager and graph drift <=5%.
Promote graph only if it is correctness-green and <= eager midpoint.

22D promoted graph:
128/1024/4096, fresh process, 12 H4 blocks.

PHASE22_TARGET_40MS_OPEN: all context medians <=40 ms/H4.
DRAFTER_SHOOTOUT_OPEN: all <=32 ms/H4.
GPU_GROUPED_MOE_PHASE23_OPEN: graph green, target40 false, Phase21 M1>=0.50.
S100_SINGLE_ACHIEVED=false.
