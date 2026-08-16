# E100-MRHS256 preregistration — one reference thread per virtual tid, up to N=16

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: **no E100-MRHS target-GPU data exists yet**.

## Why this is a separate preregistered arm

`E100_MRHS_PREREGISTRATION.md` froze an ERVF-like 32-lane-per-row geometry: each lane emulates eight of the production kernel's 256 virtual threads. That is reasonable for N=2/4, but its register footprint scales as `8*N` accumulators per physical lane.

MRHS256 tests a different, simpler exact geometry that is especially relevant to larger aggregate batches: one 256-thread block owns one output row and **physical thread tid is exactly production virtual tid tid**. Each thread therefore holds only N RHS accumulators while a loaded weight/dequantised scalar is reused across those N accumulators.

This arm is frozen before either geometry is run on the target. It cannot be justified using a later failure/success of MRHS32.

## Exact arithmetic contract

For each RHS independently:

- identical production virtual-thread assignment (`tid`, stride 256);
- identical per-thread scalar/uchar4 MAC order;
- identical first warp butterfly `16,8,4,2,1`;
- identical eight warp sums;
- identical second warp-0 butterfly `16,8,4,2,1` with lanes 8..31 zero.

The only new action is that a weight scalar decoded by tid is reused for RHS 0..N-1 before tid advances to its next production work item. Operations on other RHS accumulators do not alter a given RHS's floating-point state.

## Frozen N values

- N=4: cross-check against the smaller-batch E100 path.
- N=8: diagnostic bridge.
- **N=16: primary aggregate-throughput arm.**

N=16 is selected before target data because the existing route-union census already shows a structurally different regime: mean routed union 63.90 experts from 96 route positions (66.6% of no-overlap) versus N=4's 21.665/24 (90.3%). A future full E100 architecture can exploit more repeated routed experts at N=16; common-weight MRHS must therefore be able to support N=16 without pathological register spill.

## Real checkpoint families

At minimum:

- attention Q and O BF16;
- router F32;
- Mamba input projection in stored kind;
- Mamba output projection in stored kind;
- shared expert up/down NVFP4;
- LM head NVFP4.

K/V attention projections are included when present. Unsupported storage kinds are reported explicitly. LM head, Mamba input and Mamba output are mandatory.

## Correctness

Full mode: at least three deterministic X batches per supported case/N. Every output float for every RHS must be bit-identical to N sequential calls through the adopted exact baseline kernel; a second candidate execution must also be bit-identical. NaN/Inf closes the gate.

## Timing

CUDA-event device timing, fixed `REF, MRHS256, MRHS256, REF` ordering after warmup. Full mode >=10 repeats and >=4 rounds per arm. Raw samples retained.

Primary N=16 gates for an E100-useful common-weight primitive:

- all exactness/determinism/finite gates pass;
- weighted registered common-matrix aggregate speedup >= **3.0x**;
- LM head speedup >= **3.0x**;
- Mamba input speedup >= **2.5x**;
- Mamba output speedup >= **2.5x**;
- no supported N=16 family slower than 0.95x;
- every N=16 reference A/B drift <=7%.

Cross-check N=4 gate: weighted aggregate speedup >=1.50x. N=8 is diagnostic only.

These are component gates, not an E100 model claim. If they fail, a later routed-expert result cannot retroactively make MRHS256 'pass'.

## Full-model implication if it passes

N=16 aggregate 100 tok/s permits a 160 ms batch tick, but the runtime must remain exact and report per-sequence latency. The larger N is useful only if the full routed path can exploit its measured route overlap and if state/graph VRAM fits. No component speedup is projected into that runtime; a full integrated A/B is mandatory.
