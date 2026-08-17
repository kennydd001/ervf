# S100 phase 6 — confirmatory sparsity frontier and WAVE downflow

Date frozen: 2026-08-17

## Quality arms

Primary confirmatory arm: QFAST with K6 and global relative ReLU2 threshold `alpha=0.0003`.

Nested K arms are reconstructed from the already-saved phase-5 greedy order:

- budget 1: layer 40 -> K5;
- budget 2: layers 40 and 34 -> K5;
- budget 3: layers 40, 34 and 49 -> K5.

Combination arms add alpha 0.0003 to each nested K budget. No layer ranking or alpha is recomputed.

All arms run on validation. Alpha 0.0003 always proceeds to untouched heldout because it passed every frozen official phase-5 validation gate. The largest K budget with `strict_pass`, and the largest combination with official pass plus p95 KL <=0.075, also proceed. Every heldout result is reported; timing opens only for heldout official-pass arms.

## Exact CUDA arms

- `prefix_exact`: deterministic prefix scan plus the existing two-mirror B3 pipeline;
- `wave2_exact`: existing exact scan plus 2+2+2 routed-slot waves;
- `wave3_exact`: existing exact scan plus 3+3 waves;
- `wave6_exact`: existing exact scan plus all six slots in one wave;
- `wave3_prefix_exact`: composition of the new prefix scan and W3.

This separation is frozen before timing. A slow scan cannot falsely close WAVE, and a fast WAVE cannot hide a scan regression.

WAVE uses separate mirrors per slot, one batched gather per wave, one 3-D masked-down kernel per wave, separate per-slot partials and the existing route-ordered final accumulation.

## Exactness preflight

- prefix scan arrays equal the old scan byte-for-byte;
- threshold scan equals an independent CPU reference;
- W3 QFAST tokens equal existing QFAST on every smoke token;
- finite output and deterministic repeat;
- a forced bad route must diverge.

## Fresh timing

Every candidate runs in four independent processes: BASE_A, CAND_A, CAND_B and BASE_B. Full mode requires at least 765 samples, <=1 ms baseline and candidate drift, <=7.8 GiB VRAM and deterministic candidate A/B. Exact kernel candidates additionally require candidate-vs-base token parity.

## Telemetry

Separate instrumented validation arms record panel count, active column count and estimated host code bytes for alpha 0 and alpha 0.0003. These counts are not promoted to latency.

## Success

S100-single requires <=10.000 ms per useful token plus exact QFAST inheritance or heldout V18-fidelity green. A component benchmark or projected saving is not success.
