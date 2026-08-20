# Phase27 external CUDA notes

Checked against NVIDIA CUDA documentation in August 2026.

- CUDA devices with `asyncEngineCount > 0` can overlap asynchronous host/device
  copies with kernel execution; host memory must be page-locked and work must
  be placed in appropriate non-default streams.
- CUDA 13.x exposes `cudaMemcpyBatchAsync`, which can issue a batch of
  pointer-to-pointer copies asynchronously.
- That API is not the first Phase27 arm because LightningStream's sparse down
  source addresses are generated dynamically on device from route IDs and
  activation masks inside the captured graph. A host-side pointer list would
  require exposing those GPU results to the CPU or updating graph copy nodes
  every replay, which would violate the current no-host-sync graph contract.
- Phase27 therefore first uses GPU-resident range descriptors and concurrent
  gather/down kernels. If that route is insufficient, a later phase can
  investigate a device-driven sparse DMA/TMA representation or fused
  mapped-host zero-copy grouped down kernel.

References:
- https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-programming-guide/
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
