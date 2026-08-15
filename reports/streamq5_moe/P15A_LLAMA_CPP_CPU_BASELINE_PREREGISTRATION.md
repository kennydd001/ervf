# P15A llama.cpp same-machine CPU baseline — preregistratie

Datum: 2026-08-12

## Vraag

Hoe snel draait dezelfde lokale `Qwen3-30B-A3B-base` checkpoint als een
publiek reproduceerbare `llama.cpp` Q5_K_M-GGUF op dezelfde machine, zonder
GPU-offload, bij batch 1 en 4096 prompttokens?

Dit is een externe runtime-ankering, geen appels-met-appels bitexacte
vergelijking: de onderzoeksruntime gebruikt eigen symmetrische Q5-groepen plus
een Q8-trunk; `llama.cpp` gebruikt zijn eigen Q5_K_M-layout en kernels.

## Vergrendelde uitvoering

- broncheckpoint: `models/qwen3-30b-a3b-base`
- `llama.cpp` commit: `9558fa44c92746a58dd07ad1bf0c889715b938a6`
- build: WSL Ubuntu 24.04, Release, `GGML_CUDA=OFF`, `GGML_NATIVE=ON`
- conversie: HF BF16 -> GGUF BF16 -> Q5_K_M
- benchmark: `llama-bench`, 16 threads, `-ngl 0`, batch 1 decode,
  4096 prompttokens, 128 gegenereerde tokens, drie herhalingen
- uitvoer: JSON plus ruwe commandoregel, commit, bestandsgrootte en SHA-256

## Vooraf vastgelegde interpretatie

- De verhouding gebruikt de externe `llama.cpp` mediane decode-tok/s en de
  P13C mediane/gerapporteerde 14,23476 tok/s alleen als same-machine
  praktijksignaal.
- Een winst is interessant indien P13C minstens 1,25× meer decode-tok/s haalt.
- Geen uitkomst bewijst universele SOTA, omdat quantisatie, CPU/GPU-pad en
  runtimesemantiek verschillen.
- Falen door converter- of modelincompatibiliteit wordt als
  `blocked_runtime` geregistreerd, niet als snelheidswinst.

