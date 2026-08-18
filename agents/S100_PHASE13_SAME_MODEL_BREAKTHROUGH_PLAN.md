# S100 Phase 13 — same-model breakthrough search

Date: 2026-08-18
Parent: `agent/s100-phase12c-hardware`
Constraint: **the exact same Nemotron 3 Nano checkpoint remains the target.** No Elastic, no smaller checkpoint, no replacement model.

## What Phase 12C actually falsified

Phase 12C closes only these implementations:

- bit-exact multi-row ERVF with B independent virtual reduction trees;
- the tested grouped routed-up/down kernels at the measured shallow M distribution.

It does not prove that multi-token verification, tensor-core mini-prefill, compressed exact weights, activation-subspace execution or expert-basis compilation are impossible.

Measured reasons for failure:

- dense ERVF-M is exact but scales only 1.16x / 1.27x / 1.44x at B=2/4/8;
- grouped MoE has a 26% M=1 penalty and 0.94x weighted B=4 speedup;
- the projected perfect-draft B=4 cycle remains about 60.9 ms.

The likely mechanism is register pressure/occupancy loss from preserving B exact accumulator forests, plus insufficient rows per unique expert (mean 1.43 at B=4).

## New central insight

Stop requiring every internal kernel to be bit-identical when the project already accepts a frozen end-to-end fidelity gate. Use the same weights and architecture, but compile the execution into a fast approximate main path plus selective exact correction/fallback.

Three independent ways to reduce physical bytes are now tested.

## Track A — native tensor-core block verifier

Use native SM120 BF16/FP8/NVFP4 small-M GEMM instead of exact ERVF-M. This deliberately permits different floating-point reduction order while keeping the same checkpoint weights.

- B={2,4,8};
- real-weight cold rotation >4x L2;
- CUTLASS/cuBLASLt/PyTorch native paths;
- compare output error, route agreement, token agreement and full validation fidelity;
- candidate block state remains internal to the candidate runtime; no claim of bit identity.

A block candidate opens end-to-end integration when:

- B=4 dense useful-row speedup >=2.5x;
- B=4 routed expert path >=1.15x or is left on ERVF;
- validation official gates pass;
- projected perfect-draft B=4 cycle <=35 ms.

This tests whether the Phase-12C failure was an exact-reduction tax rather than a physical hardware limit.

## Track B — Subspace-Residual ERVF (SR-ERVF)

For each dense linear input distribution, learn a calibration-only orthonormal activation basis U_r. Compile:

`W x = (W U_r)(U_r^T x) + W r`, where `r = x - U_r U_r^T x`.

The main path reads `W U_r`, not W. The residual is handled by one of:

- skip when a frozen residual/output bound is below threshold;
- sparse top-k residual columns;
- exact ERVF residual fallback;
- periodic exact refresh.

Frozen ranks: r={128,256,384,512,768,1024}. Frozen residual gates target exact fallback rates of 0%, 10%, 25%, 50% and 100%.

This is activation-manifold compilation of the same model, not a replacement model. Selection uses `_01`; validation `_02`; heldout `_03/_04` remains untouched until frozen selection.

Promotion requires:

- >=35% measured dense-byte reduction averaged over validation;
- official validation gates;
- >=1.5 ms/token fresh end-to-end gain;
- deterministic repeat and finite state.

## Track C — Lossless Entropy-ERVF

Compress the actual FP8/BF16 dense weight bitstreams losslessly and decode inside the GEMV tile. Every decoded code/bit is exactly the original checkpoint value, so the existing FMA/reduction order can remain exact.

Evaluate:

- per-tile Shannon entropy;
- 4/5/6-bit local palettes plus escape stream;
- exponent/mantissa split coding;
- ANS/rANS tile coding;
- fused decode + ERVF versus raw ERVF under >4x L2 rotation.

First gate:

- Mamba FP8 total encoded bits <=6.0 bits/weight;
- all resident dense weights <=70% of current bytes;
- decode overhead leaves >=1.20x cold-stream speedup.

This is the safest exact same-model route and can compose with every later approximation.

## Track D — expert shared-basis residual compiler

Across the 128 routed experts of each MoE layer, fit a common basis/barycenter plus expert-specific residual:

`W_e = sum_r a[e,r] B_r + R_e`.

Test shared-rank/basis counts {4,8,16,32} and residual budgets {6.25%,12.5%,25%,50%}. Up and down are evaluated separately because ReLU2 prevents naïve pre-activation expert merging.

The runtime must never reconstruct full experts. Shared basis weights are VRAM-resident; only compact coefficients/residuals vary by expert.

Promotion requires >=30% routed-weight bytes removed and frozen fidelity green.

## Decision order

1. Run entropy and activation-manifold diagnostics first; they are cheap and decide whether physical byte reduction exists.
2. Run native tensor-core block ceiling using real weights, without drafter training.
3. Integrate only the strongest main-path candidate.
4. Train or build a drafter only after the target verifier/projected runtime clears a realistic break-even gate.
5. Explore expert-basis compilation in parallel because it attacks a separate byte pool.

## S100 claim

S100 remains <=10.000 ms/useful single-stream token or >100 accepted target tokens/s over complete wall time, with the same checkpoint and frozen heldout quality green. Component projections never count.
