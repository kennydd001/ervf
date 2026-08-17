# S100 phase 8 preregistration

## Live model contract

Discover `hidden`, `moe_inter`, expert count, top-k and MoE layers from the running QFAST runtime. Do not reuse stale Qwen/Kimi dimensions.

## Route/cache census

Record actual `need[]` cache misses and expert IDs for >=512 causal tokens. Report miss distribution, temporal route overlap and offline LRU curves.

## Real routed-down sample

Export early/middle/late layer samples containing six real panel-major NVFP4 down records, real ReLU² activations, route weights, masks and global scales. Raw samples stay local.

## Arc tests

1. same-weight M={1,2,4,6};
2. N={1,2,4,6} distinct down matrices with weighted reduction;
3. N distinct full FP16 experts;
4. custom OpenCL kernel over the actual panel-major NVFP4 records;
5. CUDA-pinned host pages wrapped with Intel OpenCL `USE_HOST_PTR`;
6. QFAST under sustained Arc load;
7. D3D12 cross-adapter shared heap/resource/fence capability.

## Promotion

Arc routed-down integration opens only if strict real-NVFP4 N=6 correctness is green, kernel plus measured bridge is <=0.25 ms/layer, and Arc contention does not erase the expected gain. Component microbenchmarks never count as S100.
