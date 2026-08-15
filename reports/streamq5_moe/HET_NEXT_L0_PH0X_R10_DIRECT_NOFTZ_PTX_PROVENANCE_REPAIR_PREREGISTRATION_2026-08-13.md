# HET-NEXT L0 PH0X-R10 — direct no-FTZ PTX provenance repair

Date: 2026-08-13. Exploratory NVIDIA completion only; no formal PH0 pass.

R10 binds PH0X-R9 runner SHA `65bc5d13aaa07689dbdc794f1735cf39310fbdf681f0269c8ee26ca6391217ae` and prereg SHA `6b35c59a34da914baaefd8103846851d67ac7e0a26bb5728bdced9aee13ec213`. It adds the exact five dependencies missing from the R9 audit before any payload/device call:

- R5 runner `0f2d1894067c65fd40200c45d7b8d14dd72d35987e7aad4afed8c003dead9f63`;
- R5 prereg `0824989026f32cb692001b4824937ad636f8a72cfd2f263e71b191d4c196aa71`;
- R6 runner `a1369c314a4e1367fa4ce3584555a7dc4db30ed9480cbdff289aa18af8417bdf`;
- R6 prereg `7e5c0ad01797120c66ce140f32207ed3460821aa3a0f4acbd6aff8f5a8231732`;
- R7 prereg `3fd0c0429eaaebe1291d9ffbbb31d08df38de300687ff3c63def1b28d0b3eb95`.

No scientific, arithmetic, kernel, PTX, buffer, launch, lifecycle, gate, threshold, identity or claim change is allowed. R10 invokes the frozen R9 main path after redirecting only its output/prereg identities to new create-only paths and after the expanded provenance gate. The CUDA driver may JIT-load the frozen textual PTX; no NVRTC source compilation occurs.

One new clean NVIDIA-only attempt; no Intel API, retry or retuning. Claim remains one real projection/input only, with no full expert/layer/model/performance/concurrency/deployment/novelty/breakthrough claim.
