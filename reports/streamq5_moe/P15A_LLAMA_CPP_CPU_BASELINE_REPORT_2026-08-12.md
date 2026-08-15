# P15A — externe llama.cpp-baseline op dezelfde machine

## Uitkomst

De actuele `llama.cpp`-commit `9558fa44` draaide dezelfde Qwen3-30B-A3B-
checkpoint als Q5_K_M, CPU-only met 16 threads. Bij 4096 prompttokens en 128
decode-tokens waren de gemiddelden:

| pad | prestatie |
|---|---:|
| promptverwerking | 39,747 tok/s |
| autoregressieve decode | 0,225149 tok/s |
| STREAMQ5 P13C decode | 14,234758 tok/s |

P13C is daarmee 63,22× sneller dan deze CPU-only ankerbaseline. De vooraf
vastgelegde 1,25×-poort is gehaald.

## Grens van het bewijs

Dit is geen geldige claim tegen de beste bestaande hybride runtime. WSL had
geen `nvcc`; de gebouwde `llama.cpp` is expliciet `GGML_CUDA=OFF`. Bovendien
verschillen Q5_K_M en de eigen Q5/INT8-semantiek. De uitkomst bewijst alleen dat
de huidige GPU-streamingruntime op deze laptop veel sneller is dan een
reproduceerbare CPU-only publieke runtime met dezelfde broncheckpoint.

Ruwe data: `p15a_llama_cpp_cpu_baseline.json`.
