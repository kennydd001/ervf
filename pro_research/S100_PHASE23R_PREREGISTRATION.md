# Phase23R preregistration

Thermal primer:
parent graph, context1024, canonical prefill, 32 H4 cycles.
Never included in A/B statistics.

Four balanced fresh-process rounds:
R1 P->G
R2 G->P
R3 G->P
R4 P->G

Each scored process:
- context1024
- 8 warmup H4 blocks
- 16 measured H4 blocks
- exact tokens
- identical canonical positions.

Statistics:
gain_round = 1 - median(grouped)/median(parent)
gain_pair  = 1 - grouped_block_ms/parent_block_ms at matched position
robust_cv  = 1.4826*MAD(process_medians)/median(process_medians)

Promote grouped iff:
- all correctness green
- median round gain >=0.05
- median paired-block gain >=0.05
- >=3/4 round gains >0
- parent robust_cv <=0.05
- grouped robust_cv <=0.05

Do not lower the 5% gate.

If promoted:
128/1024/4096, fresh grouped process, 12 H4 blocks.
No 4096 warmup because the frozen trace ends at 4145.

S100_SINGLE_ACHIEVED=false.
