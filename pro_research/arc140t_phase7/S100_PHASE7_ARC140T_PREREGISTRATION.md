# S100 Phase 7 Arc 140T preregistration

## Rules

- Never add RTX and Arc nominal TOPS into a tok/s prediction.
- CPU RAM and Arc UMA are one physical memory-capacity pool.
- Report prefill and token-generation separately.
- Preserve exact command, model hash, llama.cpp commit, device order and
  placement for every row.
- Minimum llama-bench repetitions: 5.
- Correctness smoke before a hybrid placement is called usable.
- Cross-vendor tensor mode is experimental and is never a deployment baseline.
- Static fine-grained expert placement is not promoted until contiguous layer
  split is measured.
- A microkernel or synthetic OpenVINO projection result is a geometry result,
  not an ERVF tok/s claim.
- A new S100 record requires the existing frozen model-quality protocol.

## Success levels

A. Arc endpoint characterized on the exact laptop.
B. Hybrid layer split beats both relevant endpoint/capacity baselines.
C. A topology-aware role (cold expert/coalescer/context/draft) has a positive
   measured break-even after transfers/synchronization.
D. Integrated ERVF+Arc implementation improves quality-green single-stream
   latency.
E. <=10.000 ms/useful token with quality green.
