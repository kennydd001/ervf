# HET-NEXT-L0-PH1 CPU-freeze R1 repair

Date: 2026-08-13  
State at freeze: source only; execution closed pending independent source audit.

This revision preserves every arithmetic operation, input, source range, codec rule, LUT definition and `rel_l2 <= 0.08` quality threshold in the PH1 CPU-freezer SHA-256 `746a879192041dee32acb1bcb9360ce9dde6775631c0a0671312660fb71437c8`. It changes lifecycle and provenance only:

1. Before the first payload-range read, require the exact PH1 preregistration SHA-256 `c464be6643f0301ea9f99b0e69141959a53667fa7cf9915bd540cea0a15b2b39`, design SHA-256 `4fa8a9f17b5d6c16d92c6ff1816ceda7e213e852e7fac5bfe5761c3c0338bbaf`, base freezer hash above, Qwen3-Next modeling source `de40823607becdd616436e3b332f14e0c92df5495ac72ef8af027c4488b9afca`, Transformers activations `5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665`, and dependency lock `1d08457aded09f139d25af84ba778d8e275ab5ff71967a3dc8b9a7452e6d2fae`.
2. Before the first payload read require Python `3.12.10`, Torch `2.12.1+cu132`, NumPy `2.2.6`, safetensors `0.8.0`, psutil `7.2.2`, mpmath `1.3.0`, AVX2 CPU capability, affinity exactly logical processors 0..15, Torch threads/inter-op 1, deterministic algorithms, highest matmul precision, MKLDNN enabled, and flush-denormal false. A Torch FP32 smallest-subnormal multiply-by-one witness must retain input/output uint32 bit pattern 1.
3. Require start available host RAM at least 16 GiB before any payload read. Before commit require at least 2 GiB available and Windows process `peak_wset <= 12 GiB`.
4. At the safetensors boundary serialize `detach().clone().contiguous()` for every raw tensor, preventing shared-storage failure without changing values, dtypes or shapes.
5. Redirect the four base outputs into one fresh sibling temp package directory. Add an R1 handoff, canonical manifest and commit marker last. Fsync every file before a single same-volume atomic directory rename. A valid committed final package is immutable; stale temps are quarantined before the attempt. On incidental failure, cleanup occurs first and the complete temp attempt is renamed to a unique failed-attempt directory with an fsynced failure JSON. No partial final package is permitted.

The only authorized next action after an independent GO is one CPU-only R1 freeze. No OpenCL, CUDA, CuPy, compiler, model construction, network, or device action is authorized. A source-quality negative is a valid committed scientific negative; an incidental exception is an invalid execution and does not create device eligibility.
