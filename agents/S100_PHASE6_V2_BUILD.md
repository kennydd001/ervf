# S100 phase 6 v2 build

Frozen: 2026-08-17

Branch: `agent/s100-phase6-v2-wave-downflow`

Parent evidence:

- exact V18: about 19.60 ms/token, 51.01 tok/s;
- QFAST: 18.75165 ms/token, 53.3286 tok/s, full 10,240-token V18-fidelity green;
- phase 5 selected no candidate because selection required the additional strict gate;
- alpha 0.0003 passed every official validation gate and missed only strict p95 KL (0.062986 versus 0.060);
- budget-4 missed official p95 KL by 0.002265; nested budget 1–3 had not been validated;
- heldout `_03/_04` remained untouched.

## Phase-6 v2 pack

The one-click package contains:

1. confirmatory alpha 0.0003 heldout;
2. nested K5 budget 1–3 validation/heldout;
3. fixed alpha+budget combinations;
4. exact W2/W3/W6 sparse-downflow kernels;
5. fresh-process A/C/C/B timing and exact parity gates;
6. one final summary.

WAVE allocates one mirror per route slot, gathers a fixed group of route slots with one 2-D CUDA kernel, computes them with one 3-D masked-down kernel, and leaves per-slot partials plus route-ordered accumulation unchanged.

## Integrity

- ZIP: `3c72fdfe5ec4d277f85f396554f5538f140f66a0ca2a95e6751392388c5edbd2`
- patch: `9165ce7f774821fc03acb87bc9adff8b0d312aad5ddc0762e6753dcacfc13127`

Static validation:

- Python syntax: pass;
- PowerShell variable/parser scan: pass;
- `git apply --check`: pass;
- `--whitespace=error-all`: pass.

CUDA execution remains to be measured on the target SM120 laptop. S100-single is achieved only by `<=10.000 ms/useful token` with exact QFAST inheritance or heldout fidelity green.