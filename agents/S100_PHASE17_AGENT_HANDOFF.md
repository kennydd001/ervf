# S100 Phase 17 agent handoff

16R result:
- native BF16 subset route closed;
- exact-state attention block route did not pass;
- Mamba affine SSM scan passed on layers 0/25/50 with state/output NRMSE
  around 1e-7.

Phase 17 therefore changes target. It does not approximate weights.

The block candidates consume exactly the same:
- Mamba weights;
- exact captured layer inputs;
- initial conv state;
- initial SSM state;
- dt/A/B/C/D values implied by those inputs.

Only execution order changes.

Two SSM implementations are compared:
- `prefix`: H CUDA lanes per state scalar and inclusive affine subgroup scan;
- `fused_serial`: one CUDA thread per state scalar computes H recurrences
  inside one kernel.

The latter is not mathematically parallel over H, but it still removes H
separate state-kernel launches and can be faster. The research goal is useful
wall time, not ideological scan purity.

A later Phase 18 is authorized only if the full H=4 Mamba-layer ceiling is
correctness-green and >=1.10x on every sampled layer.
