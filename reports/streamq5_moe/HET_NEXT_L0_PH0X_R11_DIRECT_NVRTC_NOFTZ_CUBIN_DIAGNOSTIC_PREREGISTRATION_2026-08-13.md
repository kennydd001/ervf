# HET-NEXT L0 PH0X-R11 — direct NVRTC no-FTZ cubin diagnostic preregistration

Date: 2026-08-13. Compile-only; no module load, allocation, copy or kernel launch.

R10 result SHA `14eb1b20b8b3f077fe5bcd73e652fe0aa4b2b6233530b637d33d73388977e51e` is a valid pre-launch negative: the CUDA driver rejected textual PTX version 9.3 as unsupported. No kernel/alloc/copy occurred; identity and cleanup evidence were retained.

R11 uses direct installed NVRTC 13.3 on the exact R8/R9 source with options `--std=c++17`, `--fmad=true`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`, `--gpu-architecture=sm_120`, and `--device-as-default-execution-space`. It retrieves cubin with `nvrtcGetCUBIN`, retains it create-only, records compiler log/hash/size, destroys the program, and performs no driver module load or device work.

Diagnostic pass requires a nonempty ELF cubin, empty/retained build log, clean program destruction, exact source/options bindings, and absence of any device execution. The already retained no-FTZ textual PTX proves the same source/options lower without `.ftz`; R11 does not claim to independently disassemble cubin. Passing may open a new separately audited cubin-load execution.
