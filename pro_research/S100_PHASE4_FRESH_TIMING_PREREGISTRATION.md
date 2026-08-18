
# S100 phase 4 — fresh-process timing preregistration

Date frozen: 2026-08-17

## Why phase 3 timing is exploratory only

The phase-3 K profiles restored `top_k`, dense weights, cache buffers and the
H-SCALE+B3 closure inside one long-lived CUDA process. Candidate repeats were
stable, and exact token parity returned, but `BASE_B` slowed by 1.7–13.4
ms/token. The arithmetic restoration worked; the performance state did not.
This makes midpoint savings invalid.

Phase 4 never restores an arm in-process. Every arm starts from a new Python
process and a newly constructed V18 runtime.

## Profiles

- `qfast`: six attention-Q matrices to NVFP4 CEIL.
- `mamba`: every non-NVFP4 Mamba in/out projection to NVFP4 CEIL.
- `fast`: Mamba plus attention-Q NVFP4 CEIL.
- `k5`: exact checkpoint, routed top-k=5.
- `k4`: exact checkpoint, routed top-k=4.
- `fast_k5`
- `fast_k4`

## Frozen process order

For each candidate, all four are independent processes:

1. `EXACT_A`
2. `CAND_A`
3. `CAND_B`
4. `EXACT_B`

Each process:

- checks the WDDM-aware idle gate before importing CuPy;
- constructs V18 from the checkpoint;
- applies the profile before cache allocation and graph capture;
- preheats 48 tokens in smoke or 128 tokens in full;
- measures the same registered prompt set;
- records all raw timings, token ids, clocks, power and VRAM.

## Gates

- exact A/B token parity;
- candidate A/B token parity;
- candidate finite;
- exact A/B p50 drift <=1.0 ms;
- candidate A/B p50 drift <=1.0 ms;
- every full arm has at least 765 samples;
- all arms <=7.8 GiB;
- runtime profile and top-k match the frozen profile.

Confirmatory information thresholds, frozen after the exploratory phase-3 run:

- qfast: gain >=0.40 ms;
- mamba: gain >=0.70 ms;
- fast: candidate <=17.80 ms;
- k5: gain >=0.30 ms;
- k4: gain >=0.60 ms;
- fast_k5: candidate <=17.60 ms;
- fast_k4: candidate <=17.20 ms.

These thresholds classify the engineering value of an arm. They are not
quality gates and not S100 claims.
