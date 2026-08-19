# S100 Lightning Phase 15 preregistration

Date: 2026-08-19

## Identity

The runner must record hashes of `config.json` and
`model.safetensors.index.json` and assert:

- max_position_embeddings = 1,048,576;
- hidden_size = 2688;
- num_hidden_layers = 52;
- n_routed_experts = 128;
- num_experts_per_tok = 6;
- moe_intermediate_size = 1856;
- vocab_size = 131072.

Failure is a technical identity failure, not a model result.

## Fresh parent trace

The trace target is the current Lightning QFAST + alpha=0.0003 parent in eager
mode. Split lengths:

- calibration `_01`: 64 targets per prompt;
- validation `_02`: 128;
- heldout `_03/_04`: 256.

Top-64 probabilities and target log-probabilities are frozen. Metadata contains
the Lightning identity and trace hash. The parent self-control must have
top-1=1.0 and negligible CE/KL.

## Precision ladder

- `round_ervf`: BF16-round activation, current ERVF, FP32 output;
- `tc1`: one BF16 activation term, FP32 accumulation/output;
- `tc2`: high+residual BF16 terms, one GEMM, FP32 sum;
- `tc3`: three BF16 terms, one GEMM, FP32 sum.

Family sets: K, V, O, KV, KO, VO, KVO.

## Cold stream

Every live Lightning BF16 matrix is executed exactly once in layer order before
the next repetition. No per-matrix timing loop. A/C/C/B order. B={1,2,4,8};
terms={1,2,3}. Inputs/outputs are preallocated.

Performance gate for terms=2, B=4:
- useful-row speedup >=2.5x;
- physically plausible aggregate bandwidth;
- max matrix output NRMSE <=0.005;
- finite.

## Quality

Calibration uses only the new Lightning calibration trace. Select at most three
candidates by quality, physical speed and family coverage.

Strict validation gates:
- top1 >=0.970;
- top5 >=0.999;
- mean CE delta <=0.025;
- mean coarse KL <=0.015;
- p95 coarse KL <=0.060;
- per-domain top1 >=0.90;
- per-domain mean CE delta <=0.080;
- finite.

Heldout uses the existing official gates and deterministic repeat.

## Decisions

- LIGHTNING_TRACE_PROVENANCE_GREEN
- BF16X2_COLD_STREAM_OPEN
- BF16X2_QUALITY_OPEN
- BF16X2_FAMILY_SELECTIVE_OPEN
- LIGHTNING_BLOCK_VERIFIER_RERUN_OPEN

S100 remains unclaimed.
