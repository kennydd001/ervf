# S100 phase 11 — dense-byte compiler and structural Mamba slimming

Date: 2026-08-18
Parent: `agent/s100-phase10-hypotheses`

## Evidence boundary

Phase 10 closes two implementation-only hypotheses:

- sparse routed-down panel caching adds approximately 0.75–0.82 ms/token of fixed overhead at every tested 8–48 MiB budget;
- the best exact Mamba ERVF-v2 stream kernel is 6.0% faster in isolation but only 0.012 ms/token faster end-to-end.

The next lever is therefore not faster fetching or streaming. It is reducing the number of dense weight bytes and state channels consumed per token while retaining the frozen fidelity gates.

## Track A — residual-corrected mixed-precision weights

For each currently FP8 Mamba in/out matrix, freeze these formats before heldout:

1. FP8 baseline;
2. NVFP4 CEIL;
3. NVFP4 CEIL plus sparse FP8 residual blocks;
4. NVFP4 CEIL plus rank-r residual correction;
5. block-mixed FP8/NVFP4, with sensitive blocks retained in FP8.

Residual budgets are fixed at 3.125%, 6.25%, 12.5% and 25% of FP8 bytes. Low-rank corrections use r in {8,16,32,64}. Block sensitivity is selected only on calibration using activation-weighted reconstruction error and output-logit influence.

The fused runtime must read the compressed main weight and correction without reconstructing a full FP8 matrix. Report physical bytes read, cold-stream latency and end-to-end latency separately.

## Track B — structured Mamba channel slimming

Prune complete Mamba channels consistently across:

- in-projection output rows;
- convolution/state parameters;
- gate/state branches;
- out-projection input columns.

Frozen retained-width fractions: 93.75%, 87.5%, 75% and 62.5%. Rank channels by calibration activation energy, state influence and output reconstruction loss. Recovery arms:

- no recovery control;
- least-squares adjacent-matrix reconstruction;
- small LoRA/distillation recovery.

This creates an explicit compiled derivative and must be named accordingly.

## Data split

- calibration: `_01` prompts;
- validation: `_02` prompts;
- heldout: `_03/_04`, never read during selection;
- external task evaluation only after heldout passes.

## Quality gates

Reuse the frozen V18/QFAST fidelity gates. No gate relaxation. In addition:

- no domain CE loss above the existing per-domain gate;
- deterministic repeat;
- no NaN/Inf;
- destructive control must fail;
- candidate artifact records exact changed matrices/channels and physical bytes.

## Performance gates

A format or structural arm opens full integration only when:

- cold real-weight stream saves at least 20% physical bytes;
- representative layer output gates pass;
- full candidate saves at least 0.75 ms/token versus the current quality-green parent;
- fresh-process A/C/C/B drift <=1 ms;
- VRAM <=7.8 GiB.

Sub-0.75 ms candidates are archived but not adopted: Phase 10 shows that small isolated gains do not compose reliably.

## First-principles targets

- Mamba dense traffic: approximately 892 MB/token;
- all dense resident traffic: approximately 2,048 MB/token;
- target Phase-11 reduction: at least 300 MB/token measured physical reads;
- target integrated candidate: <=17.6 ms/token while fidelity-green.

Phase 11 is not expected to reach S100 alone. A successful byte compiler must later compose with grouped MoE or a structural layer/expert derivative. S100 remains <=10.000 ms/useful single-stream token with frozen quality green.
