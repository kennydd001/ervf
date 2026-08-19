# S100 Phase 21 preregistration

Identity hard gate:
- model_type nemotron_h
- 52 layers
- exact snapshot e8f3c7c4de75ad84fe1bcef95d38eca76214480b
- Phase20S PHASE20B_FULL_VERIFIER_OPEN=true
- Phase20B FULL_VERIFIER_CORRECTNESS_GREEN=true

Profiler:
- context 1024
- one canonical H4 block
- synchronization around families is diagnostic only, never throughput evidence.
- report Mamba/MoE/attention/norm+add/final+lm_head, by layer and total.

Hybrid selection:
- every arm fresh process
- context 1024
- 6 advancing H4 blocks after warmup
- exact canonical future tokens from Phase20B trace
- every predicted id must match canonical target.

Promote fastest correctness-green arm.
Ties within 1% prefer: v18_device_rows, v6_device_rows,
selective_grouped, current_grouped.

Full rerun:
- selected arm only
- fresh process per 128/1024/4096
- 12 advancing H4 blocks
- wall clock includes route/cache work, sync and argmax.

PHASE21_TARGET_40MS_OPEN:
  <=40 ms/H4 at every context.

PHASE21_GRAPH_BUILD_OPEN:
  device-row arm green and >=10% faster than current grouped, OR profile shows
  >=20% host/launch overhead.

PHASE21_GROUPED_MOE_REPAIR_OPEN:
  grouped path loses to device rows while median route repeat >=0.20.

S100_SINGLE_ACHIEVED=false.
