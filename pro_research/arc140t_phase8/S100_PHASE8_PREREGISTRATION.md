# S100 phase 8 preregistration

## P8-A live route/cache census

Build the existing QFAST/V18 stack with capacity 72 and record for at least
512 generated tokens:

- `hidden`, `moe_inter`, `n_experts`, top-k, MoE layer count;
- actual `need[]` cache misses per route slot;
- miss distribution 0..6 by layer and token;
- route overlap with the previous token;
- offline LRU hit curves for capacities 16, 32, 48, 64, 72, 96;
- hot-set coverage.

This diagnostic is untimed and may synchronize every token.

## P8-B real down sample

Capture three representative MoE layers (early/middle/late) after real causal
execution. Export:

- six selected expert IDs;
- six actual ReLU² intermediate vectors;
- six route weights;
- actual panel masks;
- the six panel-major NVFP4 down records;
- global down scales;
- E2M1/E4M3 lookup tables.

The sample stays local and is not committed.

## P8-C Arc distinct-expert OpenVINO geometry

Use the *discovered* hidden/intermediate dimensions.

For N = 1, 2, 4, 6 compare:

- same weight, M=N;
- N distinct down matrices in one graph, route-weighted sum;
- N distinct full FP16 experts: up -> ReLU² -> down -> weighted sum.

Down proxy dtypes: FP16 and INT8. Full expert proxy: FP16.

Report inference wall-time after compile/warmup. Same-weight M=N is explicitly
not interpreted as N different experts.

## P8-D real NVFP4 Arc kernel

Use PyOpenCL on the Intel GPU. One work item owns one output row and loops
experts in route order. It reads current panel-major NVFP4 bytes directly,
skips inactive columns using actual masks, uses actual global scales and emits
one route-weighted hidden vector.

Autotune local size in {64,128,256}. Test N={1,2,4,6}. Compile both strict and
fast-relaxed-math variants. The strict arm is primary.

Correctness gates versus an independent NumPy decoder:

- finite;
- cosine >= 0.999;
- NRMSE <= 0.02;
- deterministic repeat.

Performance is reported as kernel event time and host wall time.

## P8-E CUDA-pinned / Intel-OpenCL bridge

For payloads corresponding to:

- one hidden vector;
- six routed intermediate vectors;
- six intermediate vectors + one returned hidden vector;
- 64 KiB;
- 256 KiB,

measure:

CUDA device -> CUDA pinned host buffer -> Intel OpenCL USE_HOST_PTR kernel ->
CUDA device.

No CPU memcpy is allowed in the timed region. Report median and p95.

## P8-F interference

Measure QFAST smoke:

BASE_A -> ARC_LOAD + QFAST -> BASE_B

Arc load uses the distinct-six routed-down proxy. Record QFAST p50 plus NVIDIA
clocks/power and Arc workload throughput. A sustained QFAST regression >5%
makes a latency-critical Arc role suspect unless the offloaded work saves more
than the regression.

## P8-G D3D12 capability

Attempt cross-adapter shared heap, placed buffer and shared cross-adapter fence
between the NVIDIA and Intel adapters. This is capability only; it does not
claim CUDA/OpenVINO zero-copy until those APIs import the same resource.

## Promotion

Arc routed-down integration opens only if:

1. real strict NVFP4 N=6 kernel passes correctness;
2. N=6 kernel wall time plus measured bridge is <=0.25 ms/layer;
3. QFAST interference is <=5%, or a measured overlap model still nets positive;
4. route/cache census and memory footprint are recorded.

The 0.25 ms threshold is an engineering gate for the next integration phase,
not an S100 claim.
