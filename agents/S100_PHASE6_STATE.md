# S100 phase 6 state

Date: 2026-08-17

Base: `pro-research@c839060bf8e03b34c12401b123f788399db807e5`

## Current measured record

- Exact V18: approximately 19.60 ms/token, 51.01 tok/s.
- QFAST: 18.75165 ms/token, 53.32864 tok/s.
- QFAST passed the frozen full 10,240-token V18-fidelity suite.
- S100-single remains unachieved; 8.75165 ms/token must still be removed from QFAST.

## Phase-5 result

Phase 5 completed without a technical failure but selected no candidate because its calibration selector required `strict_pass`.

This did not mean every tested arm failed the final frozen fidelity gates:

- `alpha=0.0003` passed every official validation gate: top1 0.971875, top5 1.0, mean CE delta 0.0120993, mean coarse KL 0.0115411, p95 coarse KL 0.0629859.
- It missed only the additional strict calibration p95-KL threshold of 0.060.
- `alpha=0.001` also passed the official gates but was dominated by alpha 0.0003 on mean CE, mean KL and p95 KL.
- The nested budget-4 K portfolio (layers 34, 40, 47 and 49 at K5) missed the official p95-KL threshold by 0.002265.
- Budget 1/2/3 portfolios had not been validated.
- No phase-5 arm reached the `_03/_04` heldout split; it remains untouched.

Frozen first greedy K actions:

1. layer 40 -> K5
2. layer 34 -> K5
3. layer 49 -> K5
4. layer 47 -> K5

## Phase-6 experiments

### Confirmatory quality frontier

- primary heldout arm: global ReLU2 relative threshold alpha 0.0003;
- validate nested K budgets 1, 2 and 3;
- validate alpha 0.0003 combined with those nested budgets;
- move only preregistered validation-qualified arms to untouched heldout;
- time only heldout `official_pass` arms.

### Exact prefix scan

Replace per-activation `atomicOr` plus thread-0 serial compaction with one thread per 16-column panel and deterministic shared-memory prefix scans. The emitted masks, ascending panel list and ascending column list must be byte-identical to the existing scan.

### WAVE sparse downflow

The current H-SCALE+B3 path still transitions through one sparse gather and one masked-down kernel per routed slot. WAVE allocates one mirror per slot and processes fixed waves:

- W2: 2+2+2
- W3: 3+3
- W6: 6

Each wave uses one 2-D gather kernel and one 3-D down kernel. The next wave is enqueued on the gather stream while the current wave computes. Per-slot partials and the existing route-ordered reducer remain unchanged, preserving expert and FMA order.

The effects are isolated deliberately:

- `prefix_exact`: new scan plus existing two-mirror B3 pipeline;
- `wave2_exact`, `wave3_exact`, `wave6_exact`: existing exact scan plus WAVE;
- `wave3_prefix_exact`: composition of both changes.

This prevents a slower experimental scan from falsely closing the WAVE geometry.

## Fail-closed gates

- scan arrays equal the old implementation exactly;
- threshold scan equals an independent CPU reference;
- W3 QFAST smoke trajectory equals current QFAST;
- deterministic candidate A/B;
- sabotage diverges;
- fresh BASE_A/CAND_A/CAND_B/BASE_B processes;
- full run >=765 samples, <=1 ms drift, <=7.8 GiB;
- exact kernel arms must match base token ids;
- approximate arms need untouched heldout fidelity green.

## Pack integrity

- `S100_PHASE6.patch`: `5535c32ef21c45c60d76eac5e694fe82ddce17219099cd95f67ddffd4ff7d7c0`
- `ervf_s100_phase6_oneclick.zip`: `92b477e6e51fb8bfdb72765d0bf62f52fd82dded18e1e052448367bd0a126296`

## Next architectural branch

If WAVE is neutral, stop rewriting panel scans. The next large kernel line is a verified SM120 grouped/block-scaled backend. CUTLASS has native SM120 narrow-precision primitives, but grouped NVFP4 generator/autotuning support is still uneven, so every backend needs nonuniform known values, real checkpoint weights and a sabotage arm before integration.

CUDA 13.2 `cudaMemcpyBatchAsync` is also relevant because it amortizes batch-transfer setup. It is not automatically usable here: sparse source pointers depend on device-side routing inside a captured graph. A dedicated descriptor/graph-capture preflight is required before replacing the graph-capturable SM gather kernel.
