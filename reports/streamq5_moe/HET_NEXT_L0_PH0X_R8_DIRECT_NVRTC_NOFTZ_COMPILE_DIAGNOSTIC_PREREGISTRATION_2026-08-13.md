# HET-NEXT L0 PH0X-R8 — direct NVRTC no-FTZ compile diagnostic preregistration

Date: 2026-08-13. Compile-only; no kernel launch, allocation, H2D or D2H.

PH0X-R7 immutable result SHA `314e08fc907965cf13b2af110b6a45424a9ac75ec5ec429b8f7bc7bf99fdba53` is formally negative: NVIDIA differs from the CPU/Intel strict no-FTZ oracle in 122/512 BF16 words while counters, sentinels, identity and 24-row lifecycle pass.

Local CuPy compiler source appends `-ftz=true` to RawModule options after the caller options. The generated textual PTX SHA `4917b2be21c29229a29bc879063781a38e840059513d34eba493ab2a91e05b88` contains `mul.ftz.f32`, `fma.rn.ftz.f32`, and `add.rn.ftz.f32`. A CPU bit emulator of those exact operations reproduced the stored NVIDIA output SHA `6525b36b...` for 512/512 words. These facts explain R7 but do not turn it positive.

R8 calls the locally installed NVRTC binding directly on the identical CUDA source with exact options `--std=c++17`, `--fmad=true`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`, `--gpu-architecture=compute_120`, and `--device-as-default-execution-space`. It retains textual PTX and compile log. A diagnostic pass requires zero `.ftz` instruction modifiers, positive presence of non-FTZ `mul.f32`, `fma.rn.f32`, `add.rn.f32`, the fixed subgroup-width-8 shuffle, and no kernel/device execution. This only opens a new, separately preregistered NVIDIA no-FTZ execution repair.
