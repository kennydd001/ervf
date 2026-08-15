# HET-NEXT L0 PH0X-R12 — direct no-FTZ cubin NVIDIA completion

Date: 2026-08-13. Exploratory NVIDIA completion only; no formal PH0 pass.

R12 binds the full PH0X-R10 provenance chain and immutable R10 failure SHA `14eb1b20...`. R10 remains a valid pre-launch failure because the driver rejected PTX 9.3. It also binds R11 diagnostic JSON SHA `21e2d57e85a3089cfa1c387827b636560de14c017fe316a8e7f9bf4de45bda25`, R11 compiler SHA `7ff376ab...`, R11 prereg SHA `d563f052...`, and no-FTZ sm_120 cubin SHA `660c22aec2574f12c15d8eed757433d0c9a30a1146fd27957adc96dcea6aaf57` (62,319-byte ELF).

The cubin was produced directly by installed NVRTC 13.3 from the exact frozen CUDA source using `--ftz=false`; compile log was empty and no module/device operation occurred. The paired textual no-FTZ PTX proves zero `.ftz` modifiers and the exact width-8 reduction DAG.

R12 invokes the exact frozen R10/R9/R7 NVIDIA lifecycle and scientific path, intercepting the one expected RawModule source construction and loading only the frozen cubin by path. The interceptor rejects all argument/source/options drift and is restored in `finally`. Row 0 of the exact 24-row ledger is normalized to `cubin_load` with bound SHA/bytes. No NVRTC source compilation occurs in R12 and no Intel API is opened.

Positive requires NVIDIA 512/512 BF16 words bitwise equal to unchanged CPU/Intel SHA `e8a00c17...`, counters one, sentinel overwritten, exact identity, exact ledger and clean releases. One new output directory, one attempt, no retry/retuning.

Claim boundary: one official real Q5 projection on one known natural activation reproduced across CPU, Intel host-USM and NVIDIA no-FTZ cubin only. No full expert, MoE, layer, model, held-out/generalized quality, cohabitation, concurrency, timing, performance, deployment, novelty or breakthrough claim.
