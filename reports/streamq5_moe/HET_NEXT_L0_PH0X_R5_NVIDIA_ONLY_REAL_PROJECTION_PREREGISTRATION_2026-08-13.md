# HET-NEXT L0 PH0X-R5 — NVIDIA-only real-projection completion preregistration

Date: 2026-08-13. Exploratory completion arm only; no formal PH0 pass.

## Immutable prior arm

PH0X-R3 result SHA-256 `e5fea8e2609f11dd294733645c9a4ecb08892c9d2070de33baacbd1a74b0df7c` is retained and never rerun. It records:

- official real Q5 record SHA `e3b10ab3...` and natural input SHA `5ce66a20...`;
- all nine pre-device controls positive;
- independent CPU output SHA `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867`;
- Intel Arc host-USM output: 512/512 BF16 words identical to CPU, every row counter one, sentinel overwritten, clean cleanup, PCI `0000:00:02.0`;
- failure only when Python assigned bytes through a structured memoryview to a CUDA pinned buffer, before CUDA compile/launch.

PH0X-R4 diagnostic JSON SHA-256 `43da909a23d13ba16090d26fac64d255898e988ddcbe28fb21c384b00f8eb77d` proves exact 675,840-byte `ctypes.memmove` pinned staging, byte-identical readback, successful NVRTC compilation of candidate source SHA `3ede786f3e71b76ee74f2591bde4cbb317a94f05e84bfd3ef5d64c22f6ce8435`, no kernel launch/H2D/D2H, and clean pinned release.

## R5 execution

R5 first verifies frozen dependency hashes and the full R3/R4 evidence predicates before any payload or CUDA call. It rebuilds the same official record/input and CPU oracle, requiring the frozen hashes.

Exactly one NVIDIA-only attempt then allocates pinned and device buffers for record 675,840 B, input 4,096 B, output 1,024 B, counters 2,048 B. Host staging uses only `ctypes.memmove`. Calls are exactly two memset, two H2D, one grid `(16,)`/block `(256,)` kernel, two D2H, and synchronization. The output and all counters are retained. Every allocation, stream, and pinned buffer receives a release attempt on every path. No Intel API is opened.

Positive requires 512/512 BF16 words bitwise equal to the same CPU oracle, all 512 uint32 counters exactly one, all sentinels overwritten, NVIDIA identity `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU` at `0000:01:00.0`, and clean release ledger. Otherwise negative/failure and nonzero exit. No retry.

Combined exploratory interpretation is permitted only after independent replay of R3+R5: the same one real projection/input was reproduced on CPU, Intel iGPU and NVIDIA dGPU. No full expert, MoE, layer, model, held-out/generalized quality, same-process coexistence, concurrency, timing, performance, deployment, novelty, or breakthrough claim.
