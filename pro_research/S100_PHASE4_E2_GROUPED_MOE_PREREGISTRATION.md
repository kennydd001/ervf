
# Phase 4 follow-up — E2 real grouped-MoE gate

The next aggregate gate uses real checkpoint expert weights, real routed
activations and real route-size distributions for N in {4,8,16}.

It compares:

- N independent current MoE executions;
- one expert-grouped execution;
- one shared cache/fetch for the route union;
- real per-expert M rows, including M=0/1/tail groups.

Required evidence:

- independent dequantized reference;
- nonuniform known-value preflight;
- route-order/scatter correctness;
- sabotage arm;
- >=4x-L2 cold weight rotation where applicable;
- activation quantization and grouping included in time;
- useful aggregate tokens/s, not rows/s.

API presence of `scaled_grouped_mm` is not a performance or correctness result.
