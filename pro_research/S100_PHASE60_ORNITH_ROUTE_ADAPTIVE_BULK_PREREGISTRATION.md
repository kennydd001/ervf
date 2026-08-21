# S100 Phase60: Ornith route-adaptive bulk dispatch preregistration

## Question

Can repeated H4 routes to the same expert reuse each cached NVFP4 record inside
one indirect M1-M4 launch family, while preserving the Phase59 parallelism for
different experts?

## Frozen candidate and controls

- Real Pottokao layer-20 experts 0 through 31 form the device cache bank.
- A device slot table selects expert records; no weights are copied or gathered
  during the timed candidate.
- Uniform route multiplicities M1, M2, M3 and M4 are measured. M1/M2/M4 cover
  32 assignments; M3 covers 30 assignments.
- Token IDs within one expert group are distinct, matching a top-k router.
- One exact-size CUDA kernel is compiled per multiplicity. Every matrix record
  is decoded once per expert group and applied to M input rows.
- Control is the passed Phase59 assignment-major M1 kernel with one duplicated
  cache record per assignment. Preparation and planning are outside both timed
  regions.
- Both arms launch gate, up, SwiGLU and down once per route bucket.

## Gates

1. Candidate and assignment-major control are bit-identical and deterministic.
2. All outputs are finite.
3. M1 indirect overhead is no more than 15% versus assignment-major M1.
4. M2, M3, and M4 are each faster than assignment-major M1 for the same number
   of assignments.
5. M4 speedup is at least 1.15x.
6. Kernels use zero local memory and at most 64 registers per thread.

## Boundary

This is a hot, already-planned route-bucket benchmark. Router top-k, bucket
construction, route-weight reduction, cache misses and end-to-end decoding are
not included.
